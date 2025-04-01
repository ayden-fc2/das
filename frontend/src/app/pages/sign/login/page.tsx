"use client"

// pages/login.tsx

import React, {useState} from "react";
import { TextField, Button, Typography, Box } from "@mui/material";
import Link from "next/link";
import {handleLoginProcess} from "@/app/pages/sign/utils";
import {useAuth} from "@/app/context/AuthContext";

const LoginPage: React.FC = () => {

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const authContext = useAuth()

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        await handleLoginProcess(email, password, authContext)
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
                            size={`large`}
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
                    <Typography variant="body2" className="text-gray-600">
                        Forget your password?{" "}
                        <Link href="/pages/sign/update" legacyBehavior>
                            <a className="text-blue-500 hover:underline">Update Password</a>
                        </Link>
                    </Typography>
                </Box>
            </Box>
        </div>
    );
};

export default LoginPage;
