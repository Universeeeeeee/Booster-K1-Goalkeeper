# Official virtual ZED X calibration seed

`SN45626933.conf` is a **nominal simulation model**, not a calibration of a physical camera. The serial number belongs to the official virtual camera.

The file is preserved byte for byte from `receiver_008_static100/settingsSN45626933.conf`, generated while SDK 5.0.7 received the official Stereolabs v3.2.0 ZED X stream. SDK 5.1 requires this explicit local seed for the synthetic serial instead of downloading a physical device calibration.

At SVGA (960 × 600), it specifies fx = fy = 370.8 px, principal point (480, 300), and a 120 mm stereo baseline. These are the official virtual model values used by the sender; they differ from the separate MuJoCo wide-camera approximation.

SHA-256: `a4856611b40913db85d6968430c4d0a80e296c01607ce655950f313618cafb7e`.

The G3 launcher records the declared hash. The independent verifier must hash the actual copied file and compare the opened SDK intrinsics and baseline with the rendered cameras. Supplying the file does not establish hardware calibration or imaging realism.

The file is byte-preserved, including its final blank line. Its local Git attributes prevent line-ending conversion and suppress only the blank-at-EOF whitespace warning.
