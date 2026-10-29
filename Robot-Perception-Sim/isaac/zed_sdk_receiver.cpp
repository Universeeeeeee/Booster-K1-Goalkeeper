// Standalone official SDK stream consumer. No renderer or custom-depth input.
#include <sl/Camera.hpp>
#include <atomic>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

uint64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        else out << ch;
    }
    return out.str() + '"';
}

template<class T> std::string label(const T& value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

struct Options {
    std::string ip = "127.0.0.1";
    int port = 30000, frames = 100, gpu = 0, timeout_seconds = 120;
    fs::path output;
    fs::path calibration_file;
    std::string calibration_sha256;
    sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    bool record_disparity = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout << "--ip 127.0.0.1 --port 30000 --frames 100 --output NEW_DIRECTORY --gpu 0 [--timeout-seconds 120] [--depth-mode PERFORMANCE|QUALITY|ULTRA|NEURAL_LIGHT|NEURAL|NEURAL_PLUS] [--record-disparity 0|1] [--calibration-file SNxxx.conf --calibration-sha256 SHA256]\n";
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        std::string value = argv[i];
        if (key == "--ip") o.ip = value;
        else if (key == "--output") o.output = value;
        else if (key == "--calibration-file") o.calibration_file = value;
        else if (key == "--calibration-sha256") o.calibration_sha256 = value;
        else if (key == "--depth-mode") {
            if (value == "PERFORMANCE") o.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
            else if (value == "QUALITY") o.depth_mode = sl::DEPTH_MODE::QUALITY;
            else if (value == "ULTRA") o.depth_mode = sl::DEPTH_MODE::ULTRA;
            else if (value == "NEURAL_LIGHT") o.depth_mode = sl::DEPTH_MODE::NEURAL_LIGHT;
            else if (value == "NEURAL") o.depth_mode = sl::DEPTH_MODE::NEURAL;
            else if (value == "NEURAL_PLUS") o.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS;
            else throw std::runtime_error("depth mode must be PERFORMANCE, QUALITY, ULTRA, NEURAL_LIGHT, NEURAL or NEURAL_PLUS");
        }
        else if (key == "--record-disparity") {
            if (value != "0" && value != "1") throw std::runtime_error("record-disparity must be 0 or 1");
            o.record_disparity = value == "1";
        }
        else {
            size_t used = 0;
            int number = std::stoi(value, &used);
            if (used != value.size()) throw std::runtime_error("invalid integer: " + value);
            if (key == "--port") o.port = number;
            else if (key == "--frames") o.frames = number;
            else if (key == "--gpu") o.gpu = number;
            else if (key == "--timeout-seconds") o.timeout_seconds = number;
            else throw std::runtime_error("unknown option: " + key);
        }
    }
    if (o.output.empty() || o.ip.empty() || o.port < 1 || o.port > 65535 ||
        o.frames < 1 || o.gpu < 0 || o.timeout_seconds < 1)
        throw std::runtime_error("invalid output, IP, port, frames, GPU or timeout");
    if (!o.calibration_file.empty() || !o.calibration_sha256.empty()) {
        if (!fs::is_regular_file(o.calibration_file) || o.calibration_sha256.size() != 64 ||
            !std::all_of(o.calibration_sha256.begin(), o.calibration_sha256.end(),
                         [](unsigned char ch) { return std::isxdigit(ch); }))
            throw std::runtime_error("calibration requires an existing file and a declared 64-character SHA256");
    }
    return o;
}

void check(sl::ERROR_CODE result, const std::string& operation) {
    if (result != sl::ERROR_CODE::SUCCESS)
        throw std::runtime_error(operation + ": " + label(result));
}

std::ofstream file(const fs::path& path, bool binary = false) {
    std::ofstream out(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode(0)));
    out.exceptions(std::ios::badbit | std::ios::failbit);
    return out;
}

void write_float_rows(const fs::path& path, const sl::Mat& mat) {
    if (mat.getDataType() != sl::MAT_TYPE::F32_C1)
        throw std::runtime_error("expected native F32_C1 for " + path.string());
    auto out = file(path, true);
    const char* data = reinterpret_cast<const char*>(mat.getPtr<sl::float1>(sl::MEM::CPU));
    const size_t row_bytes = mat.getWidth() * sizeof(float);
    if (!data || mat.getStepBytes(sl::MEM::CPU) < row_bytes)
        throw std::runtime_error("invalid CPU pointer/stride");
    for (size_t y = 0; y < mat.getHeight(); ++y)
        out.write(data + y * mat.getStepBytes(sl::MEM::CPU), row_bytes);
    out.close();
}

void intrinsics(std::ostream& out, const sl::CameraParameters& p) {
    out << "{\"fx\":" << p.fx << ",\"fy\":" << p.fy
        << ",\"cx\":" << p.cx << ",\"cy\":" << p.cy << '}';
}

struct Counts {
    std::atomic<int> written{0}, grab_errors{0}, duplicates{0};
};

// open_timeout_sec only applies to LIVE in SDK 5.0.7. Bound stream open/grab too.
class Watchdog {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::thread worker;
public:
    Watchdog(const Options& o, const Counts& counts) : worker([this, &o, &counts] {
        std::unique_lock<std::mutex> lock(mutex);
        if (cv.wait_for(lock, std::chrono::seconds(o.timeout_seconds), [this] { return done; })) return;
        try {
            auto out = file(o.output / "timeout.json");
            out << "{\"timed_out\":true,\"exit_code\":124,\"written_frames\":" << counts.written
                << ",\"grab_errors\":" << counts.grab_errors << ",\"duplicate_frames\":"
                << counts.duplicates << "}\n";
            out.close();
        } catch (...) {}
        std::cerr << "receiver deadline exceeded, exit 124\n";
        std::_Exit(124);
    }) {}
    ~Watchdog() {
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        cv.notify_one();
        worker.join();
    }
};

int capture(const Options& o, Counts& counts, std::ostream& log) {
    Watchdog watchdog(o, counts);
    sl::Camera camera;
    sl::InitParameters init;
    init.input.setFromStream(o.ip.c_str(), static_cast<unsigned short>(o.port));
    init.coordinate_units = sl::UNIT::METER;
    init.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init.depth_mode = o.depth_mode;
    init.depth_stabilization = 0;
    init.sdk_gpu_id = o.gpu;
    init.sdk_verbose = 1;
    const auto sdk_log = (o.output / "sdk.log").string();
    init.sdk_verbose_log_file = sdk_log.c_str();
    const auto settings = (o.output / "settings").string() + (o.calibration_file.empty() ? "" : "/");
    fs::create_directory(settings);
    if (!o.calibration_file.empty())
        fs::copy_file(o.calibration_file, fs::path(settings) / o.calibration_file.filename());
    init.optional_settings_path = settings.c_str();
    init.async_image_retrieval = false;
    init.async_grab_camera_recovery = true;
    log << "open stream " << o.ip << ':' << o.port << '\n' << std::flush;
    check(camera.open(init), "Camera.open");
    const auto actual = camera.getInitParameters();
    const auto info = camera.getCameraInformation();
    const auto& cfg = info.camera_configuration;
    const auto& calibration = cfg.calibration_parameters;
    if (actual.depth_mode != o.depth_mode || actual.coordinate_units != sl::UNIT::METER ||
        actual.coordinate_system != sl::COORDINATE_SYSTEM::IMAGE || actual.async_image_retrieval ||
        actual.sdk_gpu_id != o.gpu || info.input_type != sl::INPUT_TYPE::STREAM)
        throw std::runtime_error("opened configuration differs from required stereo stream configuration");
    sl::RuntimeParameters runtime;
    runtime.enable_depth = true;
    runtime.enable_fill_mode = false;
    runtime.confidence_threshold = 95;
    runtime.texture_confidence_threshold = 100;
    runtime.measure3D_reference_frame = sl::REFERENCE_FRAME::CAMERA;
    if (!actual.save((o.output / "sdk_init_parameters.json").string().c_str()) ||
        !runtime.save((o.output / "sdk_runtime_parameters.json").string().c_str()))
        throw std::runtime_error("could not save actual SDK parameter files");
    {
        auto meta = file(o.output / "metadata.json");
        meta << std::setprecision(12) << "{\"schema_version\":1,\"sdk_version\":"
             << quote(label(sl::Camera::getSDKVersion())) << ",\"depth_source\":\"zed_stereo_sdk\","
             << "\"depth_source_evidence\":" << quote("SDK " + label(actual.depth_mode) +
                    " with enable_depth; consumer has no renderer/custom-depth input") << ','
             << "\"sender_depth_source_verified\":false,\"raw_format\":\"float32_le_row_major\","
             << "\"depth_units\":\"m\",\"confidence_semantics\":\"SDK 0..100; lower is better\","
             << "\"png_source\":\"SDK rectified U8_C4 BGRA Mat.write\","
             << "\"frame_association_method\":\"single_grab_no_async_retrieval_timestamp_guard\","
             << "\"per_eye_sensor_capture_timestamp_available\":false,"
             << "\"mat_timestamp_semantics\":\"last data-manipulation time; not independent eye sensor capture\","
             << "\"timestamp_domain\":\"SDK IMAGE stream nanoseconds; epoch and sim/host mapping unverified\","
             << "\"calibration_input_file\":" << quote(o.calibration_file.string())
             << ",\"calibration_declared_sha256\":" << quote(o.calibration_sha256)
             << ",\"calibration_seed_relative_path\":"
             << quote(o.calibration_file.empty() ? "" : (fs::path("settings") / o.calibration_file.filename()).string())
             << ",\"calibration_hardware_verified\":false,"
             << "\"host_clock\":\"steady_clock nanoseconds; not subtractable from SDK clock\","
             << "\"requested_frames\":" << o.frames << ",\"stream_ip\":" << quote(o.ip)
             << ",\"stream_port\":" << o.port << ",\"requested_gpu_ordinal\":" << o.gpu
             << ",\"cuda_visible_devices\":" << quote(std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "")
             << ",\"opened_gpu_ordinal\":" << actual.sdk_gpu_id << ",\"depth_mode\":" << quote(label(actual.depth_mode))
             << ",\"coordinate_system\":" << quote(label(actual.coordinate_system))
             << ",\"serial_number\":" << info.serial_number << ",\"camera_model\":" << quote(label(info.camera_model))
             << ",\"width\":" << cfg.resolution.width << ",\"height\":" << cfg.resolution.height
             << ",\"reported_camera_fps\":" << cfg.fps << ",\"baseline_m\":" << calibration.getCameraBaseline()
             << ",\"calibration\":{\"left\":";
        intrinsics(meta, calibration.left_cam);
        meta << ",\"right\":";
        intrinsics(meta, calibration.right_cam);
        meta << "},\"depth_stabilization\":" << actual.depth_stabilization
             << ",\"confidence_threshold\":" << runtime.confidence_threshold
             << ",\"texture_confidence_threshold\":" << runtime.texture_confidence_threshold
             << ",\"fill_mode\":false,\"disparity_recorded\":" << (o.record_disparity ? "true" : "false")
             << ",\"timeout_seconds\":" << o.timeout_seconds << "}\n";
        meta.close();
    }
    auto frames = file(o.output / "frames.jsonl");
    frames << std::setprecision(12);
    sl::Mat left, right, depth, confidence, disparity;
    if (o.record_disparity) fs::create_directory(o.output / "disparity");
    uint64_t previous = 0;
    for (int index = 0; index < o.frames;) {
        auto error = camera.grab(runtime);
        const auto host_time = monotonic_ns();
        if (error != sl::ERROR_CODE::SUCCESS) {
            ++counts.grab_errors;
            log << host_time << " grab error " << error << '\n' << std::flush;
            if (index) throw std::runtime_error("grab failure interrupts consecutive frame sequence");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const uint64_t stamp = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
        if (!stamp || stamp <= previous) {
            ++counts.duplicates;
            throw std::runtime_error("zero, duplicate or decreasing IMAGE timestamp");
        }
        check(camera.retrieveImage(left, sl::VIEW::LEFT, sl::MEM::CPU), "retrieve left");
        check(camera.retrieveImage(right, sl::VIEW::RIGHT, sl::MEM::CPU), "retrieve right");
        check(camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU), "retrieve depth");
        check(camera.retrieveMeasure(confidence, sl::MEASURE::CONFIDENCE, sl::MEM::CPU), "retrieve confidence");
        if (o.record_disparity) {
            check(camera.retrieveMeasure(disparity, sl::MEASURE::DISPARITY, sl::MEM::CPU), "retrieve disparity");
            if (disparity.getWidth() != depth.getWidth() || disparity.getHeight() != depth.getHeight())
                throw std::runtime_error("disparity dimensions differ from depth");
        }
        const uint64_t after = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
        if (after != stamp) throw std::runtime_error("camera IMAGE timestamp changed during retrieval");
        const auto width = depth.getWidth(), height = depth.getHeight();
        if (!width || !height || width != static_cast<size_t>(cfg.resolution.width) || height != static_cast<size_t>(cfg.resolution.height) ||
            left.getWidth() != width || right.getWidth() != width || confidence.getWidth() != width ||
            left.getHeight() != height || right.getHeight() != height || confidence.getHeight() != height ||
            depth.getDataType() != sl::MAT_TYPE::F32_C1 || confidence.getDataType() != sl::MAT_TYPE::F32_C1 ||
            left.getDataType() != sl::MAT_TYPE::U8_C4 || right.getDataType() != sl::MAT_TYPE::U8_C4)
            throw std::runtime_error("invalid image/measure dimensions or native types");
        std::vector<float> valid;
        valid.reserve(width * height);
        const char* ptr = reinterpret_cast<const char*>(depth.getPtr<sl::float1>(sl::MEM::CPU));
        if (!ptr || depth.getStepBytes() < width * sizeof(float)) throw std::runtime_error("invalid depth stride");
        for (size_t y = 0; y < height; ++y) {
            const float* row = reinterpret_cast<const float*>(ptr + y * depth.getStepBytes());
            for (size_t x = 0; x < width; ++x)
                if (std::isfinite(row[x]) && row[x] > 0) valid.push_back(row[x]);
        }
        if (valid.empty()) throw std::runtime_error("frame has no finite positive SDK depth");
        const size_t middle = valid.size() / 2;
        std::nth_element(valid.begin(), valid.begin() + middle, valid.end());
        double median = valid[middle];
        if (valid.size() % 2 == 0)
            median = (median + *std::max_element(valid.begin(), valid.begin() + middle)) / 2;
        std::ostringstream name;
        name << std::setw(6) << std::setfill('0') << index;
        const std::string lp = "left/" + name.str() + ".png", rp = "right/" + name.str() + ".png";
        const std::string dp = "depth/" + name.str() + ".f32", cp = "confidence/" + name.str() + ".f32";
        check(left.write((o.output / lp).string().c_str()), "write left PNG");
        check(right.write((o.output / rp).string().c_str()), "write right PNG");
        write_float_rows(o.output / dp, depth);
        write_float_rows(o.output / cp, confidence);
        const std::string disparity_path = "disparity/" + name.str() + ".f32";
        if (o.record_disparity) write_float_rows(o.output / disparity_path, disparity);
        frames << "{\"frame_index\":" << index << ",\"sdk_timestamp_ns\":" << stamp
               << ",\"host_monotonic_ns\":" << host_time << ",\"host_saved_monotonic_ns\":" << monotonic_ns()
               << ",\"camera_timestamp_after_retrieval_ns\":" << after << ",\"width\":" << width
               << ",\"height\":" << height << ",\"valid_depth_count\":" << valid.size()
               << ",\"valid_depth_fraction\":" << double(valid.size()) / (width * height)
               << ",\"median_depth_m\":" << median << ",\"depth_stride_bytes\":" << depth.getStepBytes()
               << ",\"confidence_stride_bytes\":" << confidence.getStepBytes()
               << ",\"left_path\":" << quote(lp) << ",\"right_path\":" << quote(rp)
               << ",\"depth_path\":" << quote(dp) << ",\"confidence_path\":" << quote(cp);
        if (o.record_disparity) frames << ",\"disparity_path\":" << quote(disparity_path);
        frames << "}\n" << std::flush;
        previous = stamp;
        counts.written = ++index;
    }
    frames.close();
    camera.close();
    return 0;
}

int main(int argc, char** argv) {
    Options options;
    Counts counts;
    std::ofstream log;
    bool created = false;
    try {
        options = parse(argc, argv);
        const uint32_t one = 1;
        if (*reinterpret_cast<const unsigned char*>(&one) != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
            throw std::runtime_error("this raw format requires IEEE754 float32 little-endian host");
        // Atomic leaf creation refuses even an existing empty output directory.
        if (!fs::create_directory(options.output)) throw std::runtime_error("output already exists");
        created = true;
        log = file(options.output / "receiver.log");
        for (const char* name : {"left", "right", "depth", "confidence"}) fs::create_directory(options.output / name);
        capture(options, counts, log);
        auto summary = file(options.output / "summary.json");
        summary << "{\"completed\":true,\"written_frames\":" << counts.written
                << ",\"grab_errors_before_first_frame\":" << counts.grab_errors
                << ",\"duplicate_frames\":" << counts.duplicates << ",\"timed_out\":false}\n";
        summary.close();
        log << "completed " << counts.written << " consecutive unique SDK grabs\n" << std::flush;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (created) {
            try {
                if (log.is_open()) log << error.what() << '\n' << std::flush;
                auto summary = file(options.output / "summary.json");
                summary << "{\"completed\":false,\"error\":" << quote(error.what())
                        << ",\"written_frames\":" << counts.written << ",\"grab_errors\":" << counts.grab_errors
                        << ",\"duplicate_frames\":" << counts.duplicates << ",\"timed_out\":false}\n";
            } catch (...) {}
        }
        return 1;
    }
}
