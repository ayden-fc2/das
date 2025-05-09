"use client"

// pages/register.tsx

import React, {useEffect, useState} from "react";
import { TextField, Button, Typography, Box } from "@mui/material";
import Link from "next/link";
import {getCode, registerApi} from "@/app/api/account";
import {err, info} from "@/app/utils/alerter";
import {validateEmail} from "@/app/utils/common";
import {handleLoginProcess} from "@/app/pages/sign/utils";
import {useAuth} from "@/app/context/AuthContext";

const RegisterPage: React.FC = () => {
    /*
    验证码
     */
    const [verificationCode, setVerificationCode] = useState("");
    const [timer, setTimer] = useState(0);
    const [isSending, setIsSending] = useState(false);

    // 当 timer > 0 时启动倒计时
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (timer > 0) {
            interval = setInterval(() => {
                setTimer(prev => {
                    if (prev <= 1) {
                        clearInterval(interval);
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [timer]);

    // 发送验证码，发送成功后锁定按钮60s
    const handleGetCode = () => {
        if (!email || email.length <= 0 || !validateEmail(email)) {
            err('please enter a valid email');
            return
        }
        setIsSending(true);
        getCode(email, 0).then(res => {
            if (res.success) {
                setTimer(60);
                info('code sent to ' + email)
                return
            }
            err(res.message)
        }).catch(e=> {
            console.error(e);
            err('something went wrong');
        }).finally(()=> {
            setIsSending(false);
        })
    };

    /**
     * 注册
     */
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const authContext = useAuth()


    const handleRegister = (e: React.FormEvent) => {
        e.preventDefault();
        if (password !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }
        console.log("Registering with:", { username, email, password });
        // Add registration logic here
        registerApi(email, password, verificationCode, username).then(async res => {
            if (res.success) {
                await handleLoginProcess(email, password, authContext)
                return
            }
            err(res.message);
        }).catch(e => {
            console.error(e);
            err('Something went wrong');
        })
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
                        size={`small`}
                    />
                    <TextField
                        fullWidth
                        label="Email Address"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        variant="outlined"
                        required
                        size={`small`}
                    />
                    <TextField
                        fullWidth
                        label="Password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        variant="outlined"
                        required
                        size={`small`}
                    />
                    <TextField
                        fullWidth
                        label="Confirm Password"
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        variant="outlined"
                        required
                        size={`small`}
                    />
                    {/* 新增验证码输入和获取验证码按钮区域 */}
                    <Box className="flex items-center space-x-2">
                        <TextField
                            fullWidth
                            label="Verification Code"
                            value={verificationCode}
                            onChange={(e) => setVerificationCode(e.target.value)}
                            variant="outlined"
                            size={`small`}
                        />
                        <Button
                            variant="contained"
                            color="secondary"
                            onClick={handleGetCode}
                            disabled={timer > 0 || isSending}
                            className={`w-[160px]`}
                        >
                            {timer > 0 ? `${timer}s` : "Get Code"}
                        </Button>
                    </Box>
                    <Box className="text-center mt-6">
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            size="large"
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

export default RegisterPage;
