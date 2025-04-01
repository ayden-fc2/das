"use client"

import React from "react";
import {Box, Divider, Typography} from "@mui/material";
import Link from "next/link";

const DashBoard: React.FC = () => {
    return (
        <Box className={`h-full flex flex-col w-full`}>
            {/*Project Overview*/}
            {/*  项目标题  */}
            <Typography variant={`h4`} className={`font-bold`}>
                DCS Analysis Structurizer
            </Typography>
            <Divider />
            {/*  git地址  */}
            {/*  项目简介  */}
            <Typography className={`text-gray-700 !m-2`}>
                Intro TODO
                <Link href={``} className={`text-blue-700 underline !p-2`}>
                    github
                </Link>
            </Typography>
            {/*  README  */}
            <Box className={`w-full bg-gray-200 flex-1 p-2 rounded overflow-y-scroll overflow-x-hidden`}>
                README TODO
            </Box>
        </Box>
    );
};

export default DashBoard;
