"use client"

import React from "react";
import { Box } from "@mui/material";

const Spaces: React.FC = () => {
    return (
        <Box className={`bg-gray-300 h-full flex flex-col justify-center items-center w-full`}>
            Joined Spaces
            {/*  可筛选列表（名称、组织、创建时间、用户角色），获取到跟用户有关的所有项目空间，此处允许管理/删除/处理申请（操作列），无管理的可以退出空间/分析项目  */}
        </Box>
    );
};

export default Spaces;
