"use client"

import React, {useEffect} from "react";
import { Box } from "@mui/material";
import {navigateTo} from "@/app/utils/navigator";
import {useAuth} from "@/app/context/AuthContext";

const LogoutPage: React.FC = () => {
    const authContext = useAuth()
    useEffect(()=> {
        localStorage.removeItem("jwt");
        navigateTo("/pages/homepage")
        authContext.refreshRole()
    }, [])
    return (
        <Box className={`bg-gray-300 h-full flex flex-col justify-center items-center w-full`}>
            Joined Spaces
        </Box>
    );
};

export default LogoutPage;
