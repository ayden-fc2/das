import React from "react";
import { Container, Typography } from "@mui/material";

export default function Page() {
    return (
        <Container
            maxWidth="md"
            className="flex flex-col items-center justify-center min-h-screen bg-gray-100"
        >
            <Typography
                variant="h3"
                component="h1"
                className="text-blue-500 font-bold"
            >
                DAS Demo Page
            </Typography>
        </Container>
    );
}
