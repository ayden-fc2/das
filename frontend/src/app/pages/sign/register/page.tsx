"use client"

// pages/register.tsx

import React, {useEffect, useState} from "react";
import { TextField, Button, Typography, Box } from "@mui/material";
import Link from "next/link";
import {useAlert} from "@/app/components/AlertBanner";

const RegisterPage: React.FC = () => {
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const { showAlert } = useAlert();

    useEffect(()=>{
        showAlert('The login and registration functionality is currently unavailable. Please use the test account "1234@test.com" with the password "1234".', 'warning')
    }, [])

    const handleRegister = (e: React.FormEvent) => {
        e.preventDefault();
        if (password !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }
        console.log("Registering with:", { username, email, password });
        // Add registration logic here
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            {/* 毛玻璃容器 */}
            <Box
                className="backdrop-blur-lg bg-white/70 shadow-lg rounded-lg p-8 w-full max-w-md"
            >
                <Typography
                    variant="h4"
                    align="center"
                    gutterBottom
                    className="font-bold text-gray-800"
                >
                    Create an Account
                </Typography>
                <form onSubmit={handleRegister} className="space-y-4">
                    <TextField
                        fullWidth
                        label="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        variant="outlined"
                        required
                    />
                    <TextField
                        fullWidth
                        label="Email Address"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        variant="outlined"
                        required
                    />
                    <TextField
                        fullWidth
                        label="Password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        variant="outlined"
                        required
                    />
                    <TextField
                        fullWidth
                        label="Confirm Password"
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        variant="outlined"
                        required
                    />
                    <Box className="text-center mt-6">
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
                        >
                            Register
                        </Button>
                    </Box>
                </form>
                <Box className="mt-4 text-center">
                    <Typography variant="body2" className="text-gray-600">
                        Already have an account?{" "}
                        <Link href="/pages/sign/login" legacyBehavior>
                            <a className="text-blue-500 hover:underline">Login</a>
                        </Link>
                    </Typography>
                </Box>
            </Box>
        </div>
    );
};

export default RegisterPage;
