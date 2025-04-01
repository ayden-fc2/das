"use client"

import React from "react";
import { Box } from "@mui/material";

const NewSpace: React.FC = () => {
    return (
        <Box className={`bg-gray-300 h-full flex flex-col justify-center items-center w-full`}>
            New Space
            {/*  创建一个项目空间，支持快速拉组  */}
            {/*  可筛选列表（名称、组织、创建时间），获取到所有项目空间（有限信息），此处允许申请加入  */}
            {/*  查看申请记录  */}
        </Box>
    );
};

export default NewSpace;
