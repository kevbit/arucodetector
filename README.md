
# ArucoDetector

This project uses Conda for package management. Follow the instructions below to set up and run the project.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine.
- [Redis](https://redis.io/download) installed and running on port `6379`.

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kevbit/arucodetector.git
    cd arucodetector
    ```

2. **Create and activate a new conda environment:**

    ```bash
    conda create --name myenv python=3.10.6
    conda activate myenv
    ```

3. **Install the required packages:**

    ```bash
    conda install async-timeout==4.0.3 numpy==1.24.2 opencv-contrib-python==4.5.5.64 pyrealsense2==2.55.1.6486 redis==5.0.8
    ```

4. **Run a Redis server:**

    Ensure you have Redis installed. If not, download and install it from [here](https://redis.io/download).

    Start the Redis server on the default port `6379`:

    ```bash
    redis-server
    ```

    Alternatively, you can use a Redis service manager or run Redis in a Docker container:

    ```bash
    docker run -d -p 6379:6379 redis
    ```

## Running the Project

1. **Navigate to the project directory (if not already there):**

    ```bash
    cd arucodetector
    ```

2. **Ensure the Redis server is running:**

    Make sure the Redis server is active and listening on port `6379`.

3. **Run the main script or application:**

    ```bash
    python arucodetector.py
    ```

## Additional Notes

- Make sure your Conda environment is activated every time you work on this project.
- To deactivate the Conda environment, use:

    ```bash
    conda deactivate
    ```

## Troubleshooting

- If you encounter issues with package versions, consider updating Conda and the packages using:

    ```bash
    conda update conda
    conda update --all
    ```

- If the Redis server is not running, ensure you have started it as per the instructions above.

