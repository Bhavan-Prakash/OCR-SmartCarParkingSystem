# Car ID: Automated Vehicle Registration Monitoring For Campus

## Overview
This project is a real-time vehicle registration monitoring system designed for campus security. It leverages Jetson Nano, OpenCV, and external APIs to detect and retrieve vehicle registration details from number plates, ensuring efficient parking management and security.

## Features
- **Real-time Number Plate Detection**: Uses OpenCV for accurate license plate recognition.
- **Edge Computing with Jetson Nano**: Ensures efficient on-device processing.
- **External API Integration**: Retrieves vehicle owner data for enhanced security.
- **Supports Large-Scale Monitoring**: Capable of tracking over 200 vehicles on campus.

## Technologies Used
- **Programming Language**: Python
- **Computer Vision**: OpenCV
- **Hardware**: Jetson Nano
- **External APIs**: RapidAPI & Vehicle Registration Databases

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/car-id-monitoring.git
   cd car-id-monitoring
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt  # For Python dependencies
   ```
3. Ensure Jetson Nano is set up with OpenCV and necessary libraries.
4. Run the application:
   ```sh
   python main.py
   ```


## Contributing
Feel free to submit issues or pull requests to enhance the project.

## License
This project is open-source and available under the [MIT License](LICENSE).

