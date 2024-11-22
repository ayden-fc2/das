# DWG Structurizer

A cloud-based application that automatically structures AutoCAD DWG files into JSON format and provides an interactive interface for visualization and bidirectional adjustments.

## Project Overview

This project, built with **SpringCloud** for backend services and **Next.js** for frontend interaction, aims to streamline the extraction and organization of DWG files by converting them into a structured JSON format. The structured data is then displayed on the frontend, where users can make fine-grained adjustments that are saved back to the backend.

## Key Features

- **Automated DWG Parsing**: Reads DWG files, extracting essential data and organizing it into a JSON structure.
- **JSON Visualization**: Provides a clear visualization of DWG data in JSON format.
- **Bidirectional Data Adjustment**: Users can make adjustments to the JSON data directly in the frontend, which syncs with the backend.
- **Scalable Microservice Architecture**: Utilizes SpringCloud for modularity, with independent services for DWG parsing, JSON structuring, and data storage.
- **User Authentication and Authorization**: Protects data and enables role-based access control.

## Project Structure

- `backend/`: Contains the SpringCloud microservices.
  - `dwg-parser-service/`: Service for parsing and converting DWG files to JSON.
  - `data-storage-service/`: Manages the storage and retrieval of JSON data.
  - `auth-service/`: Handles user authentication and authorization.
- `frontend/`: Built with Next.js for rendering and managing the bidirectional JSON editor interface.
- `docs/`: Project documentation.
- `tests/`: Automated tests for backend and frontend components.

## Prerequisites

- **Java 11+** and **SpringCloud**
- **Node.js 14+** and **Next.js**
- Required dependencies in `backend/pom.xml` and `frontend/package.json`

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/dwg-structurizer.git
    cd dwg-structurizer
    ```

2. **Backend Setup**:
    ```bash
    cd backend
    mvn clean install
    mvn spring-boot:run
    ```

3. **Frontend Setup**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

4. **Configure Database**: 
   Set up your database (e.g., MySQL, MongoDB) and update connection settings in `backend/data-storage-service/src/main/resources/application.properties`.

## Usage

1. **Upload DWG File**: Upload your DWG file using the frontend interface.
2. **View JSON Structure**: The parsed DWG data will be displayed in JSON format.
3. **Edit JSON Data**: Make adjustments directly to the JSON data in the editor.
4. **Save Changes**: The adjusted JSON data will be sent back to the backend and stored.

### Example Workflow

```bash
# Backend service
cd backend/dwg-parser-service
mvn spring-boot:run

# Frontend service
cd frontend
npm run dev
```

## Contributing
Please see the issues page for open issues and contribution guidelines.

## License
Licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions, contact me at [shiqixuan@bilibili.com].