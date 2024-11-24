"use client"

// pages/login.tsx

import React, {useState} from "react";
import { TextField, Button, Typography, Box } from "@mui/material";
import Link from "next/link";
import {get} from "@/app/utils/api"
import {err, success} from "@/app/utils/alerter";
import {navigateTo} from "@/app/utils/navigator";

const LoginPage: React.FC = () => {

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        console.log("Logging in with:", { email, password });
        try {
            const response = await get<{ success: number; message: string; data: any }>(
                "/auth-service/sign/signInCheck",
                { email, password }
            );
            if (response.success === 1 && response.data) {
                localStorage.setItem("jwt", response.data);
                success("Login successful, your login status will be saved。")
                navigateTo("/pages/homepage")
            } else {
                console.error("Login failed:", response.message);
                err(response.message)
            }
        } catch (error) {
            console.error("Login Error:", error);
            err("An issue has occurred. Please try again.")
        }
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
                    Login to Your Account
                </Typography>
                <form onSubmit={handleLogin} className="space-y-4">
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
                    <Box className="text-center mt-6">
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
                        >
                            Login
                        </Button>
                    </Box>
                </form>
                <Box className="mt-4 text-center">
                    <Typography variant="body2" className="text-gray-600">
                        Don&#39;t have an account?{" "}
                        <Link href="/pages/sign/register" legacyBehavior>
                            <a className="text-blue-500 hover:underline">Sign Up</a>
                        </Link>
                    </Typography>
                </Box>
            </Box>
        </div>
    );
};

export default LoginPage;
