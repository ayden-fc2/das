"use client"

import React from "react";
import { Box, Divider, Typography } from "@mui/material";
import Link from "next/link";

const DashBoard: React.FC = () => {
    return (
        <Box className={`h-full flex flex-col w-full p-8`}>
            {/* Project Title */}
            <Typography variant={`h4`} className={`font-bold`}>
                DCS Analysis Structurizer
            </Typography>
            <Divider className={`!my-4`} />
            {/* Project Overview */}
            <Typography className={`text-gray-700 !m-2`}>
                This platform provides a comprehensive intelligent management solution for Distributed Control Systems (DCS) in the industrial field. It leverages advanced drawing parsing technology and a fault tracing prediction model to automatically parse DCS drawings, construct directed graphs from the extracted structure, and achieve high-precision fault tracing, significantly reducing the reliance on manual interpretation and expert experience.
                <Link href={`https://github.com/ayden-fc2/das.git`} className={`text-blue-700 underline !p-2`}>
                    GitHub
                </Link>
            </Typography>
            {/* README Section */}
            <Box className={`w-full bg-gray-200 flex-1 px-4 py-2 my-4 rounded overflow-y-scroll overflow-x-hidden`}>
                <Typography variant="h6" className="font-bold">Project Introduction</Typography>
                <Typography className="text-gray-700 my-2">
                    The system employs a hybrid parsing algorithm developed based on the open-source library LibreDWG. It extracts primitive elements, block information, and topological connections from raw DCS drawings, constructing a directed graph that reflects the dependency relationships among components. On this basis, a Graph Attention Network (GAT) model with bidirectional information propagation is introduced to efficiently predict and trace the source of faults.
                </Typography>
                <Typography variant="h6" className="font-bold">System Architecture</Typography>
                <Typography className="text-gray-700 my-2">
                    The platform is built on a Spring Cloud microservices architecture and utilizes Docker containerization to ensure excellent scalability, cross-platform compatibility, and high performance. In addition, the platform integrates OAuth2 authentication and a high-performance web front-end built with Next.js, delivering an end-to-end intelligent management solution from drawing parsing and fault tracing to operational management.
                </Typography>
                <Typography variant="h6" className="font-bold">Key Features</Typography>
                <ul className="list-disc ml-5 text-gray-700">
                    <li>Automated structural parsing of DCS drawings based on LibreDWG</li>
                    <li>Extraction of topological features using a rule-driven and iterative optimization approach</li>
                    <li>Fault tracing using a bidirectional information propagation Graph Attention Network (GAT) model</li>
                    <li>Spring Cloud microservices architecture with Docker containerized deployment</li>
                    <li>Layered project storage and role-based access control for industrial-level applications</li>
                </ul>
                <Typography variant="h6" className="font-bold mt-4">Usage Instructions</Typography>
                <Typography className="text-gray-700 my-2">
                    Please refer to the GitHub repository for detailed installation and usage documentation. Follow the provided steps to configure your environment, build, and start the various service modules. The provided web front-end can be used to monitor and manage the drawing parsing and fault tracing process in real time.
                </Typography>
                <Typography variant="h6" className="font-bold mt-4">Contact Us</Typography>
                <Typography className="text-gray-700 my-2">
                    If you have any questions or suggestions, please contact us via GitHub issues or email.
                </Typography>
            </Box>
        </Box>
    );
};

export default DashBoard;
