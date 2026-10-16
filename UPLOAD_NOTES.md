# Upload notes

This snapshot includes source code, documentation, robot assets, models, and all validation reports. Virtual environments, Python/test caches, operating system metadata, runtime logs, and nested Git metadata are excluded. Large data files are stored with Git LFS; run `git lfs pull` after cloning to retrieve them.

Local paths were anonymized in 16 documentation/report files: `/path/to/booster` replaces the project directory, and `/home/user` replaces the local home directory. Recorded hashes and protected manifests describe the original validation artifacts before this upload anonymization. They have not been recalculated. The original local artifacts remain unchanged.

## October 16 update

Includes the I2 independent training model and collision validation, the standing task and PPO training/evaluation pipeline, saved checkpoints, all new validation reports, an independent reproduction run, and the supplied research PDFs. Install the additional training dependencies with `pip install -r requirements-ppo.txt`.

Start with [I2 model validation](docs/training_model.md) and [standing training results](docs/standing_training.md). The current full test suite has 412 passing tests. The standing experiment trained for 16,384 steps; the reference posture, initial network, and final network each passed all five fixed evaluation cases. These results do not establish general goalkeeper or broad disturbance-rejection capability.

The same upload-only local-path anonymization applies to this update. Original artifact hashes, checkpoint hashes, validation manifests, and protocol source hashes refer to local validation originals. Text-file hashes may differ after anonymization; binary artifacts are copied byte-for-byte. Local source files and validation evidence remain unchanged.
