// pages/homepage.tsx
"use client";

import React from "react";
import Link from "next/link";
import { Button, Typography, Container, Box } from "@mui/material";

const HomePage: React.FC = () => {
    return (
        <Container maxWidth="md" className="flex flex-col items-center justify-center min-h-screen">
            {/* Project Title */}
            <Typography variant="h3" align="center" gutterBottom className="font-bold">
                Project Details
            </Typography>

            {/* Project Description */}
            <Typography variant="body1" align="center" className="mb-4">
                DAS (DCS Auto Structurizer) is a project dedicated to building an intelligent toolchain
                for automatically parsing and structuring DWG files based on DCS standards. By analyzing
                component layouts and upstream-downstream relationships, the project aims to improve
                the efficiency of file interaction and logical mapping.
            </Typography>

            <Box className="mt-8 text-center">
                {/* Project Progress */}
                <Typography variant="h5" align="center" className="font-bold">
                    Current Progress
                </Typography>
                <Typography variant="body1" align="center" className="mt-4">
                    Step 1.1 ✅: Built the project structure using Next.js and SpringCloud. <br />
                    Step 1.2 ✅: Implemented a robust authentication and file upload module. <br />
                    Step 2.1 ✅: Enabled test users to manage and read DWG files (using LibreDWG). <br />
                    Step 3.1 ✅: Extract the DCS components and store key information. <br />
                    Step 3.2 TODO: Developed a machine learning model to classify DCS components based on their geometries. <br />
                    Step 4.1 TODO: Analysis the upstream-downstream relationships between DCS components. <br />
                    Step 5.1 TODO: Complete the DWG file structuring process. <br />
                </Typography>
            </Box>

            {/* GitHub Address */}
            <Box className="text-center mt-8">
                <Typography variant="h6">
                    Project Source Code:
                </Typography>
                <a
                    href="https://github.com/ayden-fc2/das"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:underline"
                >
                    https://github.com/ayden-fc2/das
                </a>
            </Box>

            {/* Navigation Buttons */}
            <Box className="mt-8 text-center">
                <Link href="/pages/demo/das" passHref className="m-2">
                    <Button variant="contained" color="primary" className="bg-blue-500 hover:bg-blue-600">
                        Demo Page
                    </Button>
                </Link>
                <Link href="http://www.fivecheers.com:2072" passHref className="m-2">
                    <Button variant="contained" color="primary" className="bg-blue-500 hover:bg-blue-600">
                        SpringCloud Eureka
                    </Button>
                </Link>
                <Link href="http://www.fivecheers.com/desktop/blog?selectedKey=12-1-0-0" passHref className="m-2">
                    <Button
                        variant="outlined"
                        color="secondary"
                        className="ml-4"
                    >
                        View Documentation
                    </Button>
                </Link>
            </Box>
        </Container>
    );
};

export default HomePage;
