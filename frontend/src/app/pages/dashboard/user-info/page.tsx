"use client"

import React from "react";
import {Avatar, Box, Divider, Typography} from "@mui/material";
import {useAuth} from "@/app/context/AuthContext";
import CalendarHeatmap from 'react-calendar-heatmap';
import 'react-calendar-heatmap/dist/styles.css';

const UserInfo: React.FC = () => {
    const authContext = useAuth()
    // 计算 startDate（当前日期的 7 个月前的 1 号）
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 7);
    startDate.setDate(1);

    // 计算 endDate（下个月的 1 号）
    const endDate = new Date();
    endDate.setMonth(endDate.getMonth() + 1);
    endDate.setDate(1);

    return (
        <Box className={`h-full flex flex-col w-full`}>
            {/*  用户信息（昵称更改），信息图表统计  */}
            <Box className={`w-full h-36 p-4 bg-white shadow rounded-lg flex`}>
                <Box className={`h-full w-96 relative flex items-center justify-between`}>
                    {/*  头像  */}
                    <Avatar className={`!h-20 !w-20`} alt="Remy Sharp" src={authContext.basicUserInfo?.avatar || ''} />
                    {/*  昵称和邮箱  */}
                    <Box className={`h-full flex-1 ml-4 flex-col flex justify-center relative`}>
                        <Typography variant="h6" className={`font-bold max-w-60 truncate whitespace-nowrap overflow-hidden`}>{authContext.basicUserInfo?.nickname || 'Default Name'}</Typography>
                        <Typography variant="body2" className={`text-gray-600 truncate max-w-60 whitespace-nowrap overflow-hidden`}>{authContext.basicUserInfo?.email || ''}</Typography>
                    </Box>
                </Box>
                <Divider />
                {/*  github热力图  */}
                <Box className={`flex-1 ml-4`}>
                    <Box className={`flex items-center justify-end`}>
                        <CalendarHeatmap
                            startDate={startDate}
                            endDate={endDate}
                            values={[
                                { date: '2016-01-01', count: 12 },
                                { date: '2016-01-22', count: 122 },
                                { date: '2016-01-30', count: 38 },
                                // TODO 获取打点统计
                            ]}
                        />
                    </Box>
                </Box>
            </Box>
            {/*  我的组织以及角色，允许新建组织、管理自己的组织和角色  */}
            <Box className={`w-full flex-1 mt-4 px-4 py-2 bg-white shadow rounded-lg`}>
                <Typography variant={`h6`}>Org Info</Typography>
                {/*  筛选表格组件，组织名称、管理员、创建日期、用户角色，可选操作是管理组织弹窗  */}
                {/*  按钮弹窗新建组织（复用）  */}
                {/*  按钮弹窗审批中心，小红点  */}
            </Box>
        </Box>
    );
};

export default UserInfo;
