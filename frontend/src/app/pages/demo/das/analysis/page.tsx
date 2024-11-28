"use client";

import React, {useState} from "react";
import {navigateTo, useQueryParams} from "@/app/utils/navigator";
import { Button, Box, Typography } from "@mui/material"; // 引入Material UI组件
import { getJsonObj } from "@/app/utils/das";

interface AnalysisPageProps {
    projectName: string;
    jsonPath: string;
}

export default function AnalysisPage() {
    /*
     * 页面主体
     */
    // 初始化
    React.useEffect(() => {
        setProjectJson(getJsonObj(basicInfo.jsonPath))
    }, []);
    // 获取页面参数
    const queryParams = useQueryParams();
    const basicInfo: AnalysisPageProps = {
        projectName: queryParams['projectName'] || "未知项目",
        jsonPath: queryParams['jsonPath'] || "",
    };
    // 获取项目json
    const [projectJson, setProjectJson] = useState<object>();
    // 退出页面
    const quitPage = () => {
        navigateTo("/pages/demo/das");
    }
    // TODO: 保存更改
    const saveChanges = ()=>{
        console.log('TODO')
    }


    return (
        <div className="flex flex-col min-h-screen p-4">
            {/* 页面顶部，显示项目名称 */}
            <Box className="mb-4">
                <Typography variant="h5" component="h4" gutterBottom>
                    Project: { basicInfo.projectName}
                </Typography>
            </Box>

            {/* 主内容区域，左侧显示画布和逻辑关系，右侧显示操作面板 */}
            <Box className="flex flex-grow">
                {/* 左侧部分 - 画布渲染区域和逻辑关系展示 */}
                <Box className="flex flex-col flex-grow mr-4">
                    {/* 画布渲染区域 */}
                    <Box className="flex-grow border-2 border-gray-400 mb-4 p-4 rounded-lg">
                        <Typography variant="h6" className="mb-4">Render</Typography>
                        TODO
                    </Box>

                    {/* 逻辑关系展示区域 */}
                    <Box className="border-2 border-gray-400 p-4 rounded-lg">
                        <Typography variant="h6" className="mb-4">Logical Relationships</Typography>
                        TODO
                    </Box>
                </Box>

                {/* 右侧部分 - 操作面板 */}
                <Box className="w-64 border-2 border-gray-400 p-4 flex flex-col align-items-center rounded-lg">
                    <Typography variant="h6" className="mb-4">Coperations</Typography>
                    <div className="flex-1"></div>
                    <Button variant="contained" color="primary" className="!mb-2" onClick={saveChanges}>保存</Button>
                    <Button variant="outlined" color="primary" onClick={quitPage}>退出</Button>
                </Box>
            </Box>
        </div>
    );
}
