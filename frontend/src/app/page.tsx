// pages/index.tsx
"use client"

import React, {useEffect, useState} from 'react';
import Link from 'next/link';
import { Button, Typography, Container, Box } from '@mui/material';
import { isTokenExpired } from '@/app/utils/api'

const HomePage: React.FC = () => {

    // 初次加载页面判断token是否有效
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    useEffect(()=>{
        const token: string | null = localStorage.getItem("jwt");
        if (token){
            setIsLoggedIn(!isTokenExpired(token));
        }
    }, [])

    return (
        <Container maxWidth="md" className="flex flex-col items-center justify-center min-h-screen">
            {/* Project Title */}
            <Typography variant="h3" align="center" gutterBottom className="font-bold">
                Welcome to the DAS Project
            </Typography>

            {/* Project Description */}
            <Typography variant="body1" align="center" className="mt-4 text-gray-600">
                This project leverages advanced algorithms to automatically parse and structure DWG files
                based on DCS standards. It also visualizes logical relationships within DCS files, enabling
                users to quickly analyze, modify, and write to DWG files with greater efficiency.
            </Typography>

            {/* Login Button */}
            <Box className="mt-8 text-center">
                {isLoggedIn ? (
                    <Link href="/pages/homepage" passHref>
                        <Button variant="contained" color="success" className="bg-green-500 hover:bg-green-600">
                            Welcome Back
                        </Button>
                    </Link>
                ) : (
                    <Link href="/pages/sign/login" passHref>
                        <Button variant="contained" color="primary" className="bg-blue-500 hover:bg-blue-600">
                            Go to Login
                        </Button>
                    </Link>
                )}
            </Box>
        </Container>
    );
};

export default HomePage;
