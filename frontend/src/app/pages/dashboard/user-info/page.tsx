"use client"

import React from "react";
import {Box, Divider, Typography} from "@mui/material";

const UserInfo: React.FC = () => {
    return (
        <Box className={`h-full flex flex-col w-full`}>
            {/*  用户信息（昵称更改），信息图表统计  */}
            <Box className={`w-full h-36 px-4 py-2 bg-white shadow rounded-lg flex`}>
                <Box className={`h-full bg-amber-800 w-80`}>
                    {/*  头像  */}

                    {/*  昵称更改  */}
                </Box>
                <Divider />
                {/*  github的热力表  */}
                <Box className={`bg-amber-100 flex-1 ml-4`}>

                </Box>
            </Box>
            {/*  我的组织以及角色，允许新建组织、管理自己的组织和角色  */}
            <Box className={`w-full flex-1 mt-4 px-4 py-2 bg-white shadow rounded-lg`}>
                <Typography variant={`h6`}>Org Info</Typography>
                {/*  筛选表格组件，组织名称、管理员、创建日期、用户角色，可选操作是管理组织弹窗  */}
                {/*  按钮弹窗新建组织  */}
                {/*  按钮弹窗审批中心，小红点  */}
            </Box>
        </Box>
    );
};

export default UserInfo;
