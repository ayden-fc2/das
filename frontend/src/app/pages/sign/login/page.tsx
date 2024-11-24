"use client"

// pages/login.tsx

import React, {useEffect, useState} from "react";
import { TextField, Button, Typography, Box } from "@mui/material";
import Link from "next/link";
import {get} from "@/app/utils/api";
import { useAlert } from "@/app/components/AlertBanner";
import { useRouter } from "next/navigation";

const LoginPage: React.FC = () => {

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const router = useRouter();
    const { showAlert } = useAlert();
    useEffect(()=>{
        showAlert('The login and registration functionality is currently unavailable. Please use the test account "1234@test.com" with the password "1234".', 'warning')
    }, [])

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
                showAlert("Login successful, your login status will be saved。", "success");
                router.push("/pages/homepage")
            } else {
                console.error("Login failed:", response.message);
                showAlert(response.message, "error");
            }
        } catch (error) {
            console.error("Login Error:", error);
            showAlert("An issue has occurred. Please try again.", "error");
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
