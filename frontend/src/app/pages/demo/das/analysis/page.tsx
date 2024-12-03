"use client";

import React, {useEffect, useRef, useState} from "react";
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
    const [projectJson, setProjectJson] = useState<any>();
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [scale, setScale] = useState(1);

    useEffect(() => {
        getJsonObj(basicInfo.jsonPath).then(res=>{
            setProjectJson(res)
        })
    }, []);
    const initCanvas = (projectJson: any) => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) {

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                canvas.width = width;
                canvas.height = height;
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, width, height);

                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.translate(canvas.width / 2, canvas.height / 2);
                ctx.scale(scale, scale);
                ctx.translate(offset.x, offset.y);
                ctx.translate(offset.x / scale, offset.y / scale);

                for ( const line of projectJson?.TYPES?.LINE ?? []){
                    ctx.beginPath();
                    ctx.moveTo(line.start[0], line.start[1]);  // 起点坐标
                    ctx.lineTo(line.end[0], line.end[1]); // 终点坐标
                    ctx.strokeStyle = "#" + line.color.rgb; // 线条颜色
                    ctx.lineWidth = 2; // 线条宽度
                    ctx.stroke();
                }

                for (const circle of projectJson?.TYPES?.CIRCLE ?? []) {
                    ctx.beginPath();
                    ctx.arc(circle.center[0], circle.center[1], circle.radius, 0, 2 * Math.PI);
                    ctx.fillStyle = "#" + circle.color.rgb; // 填充颜色
                    ctx.strokeStyle = "#" + circle.color.rgb; // 边框颜色
                    ctx.lineWidth = 2; // 边框宽度
                    ctx.stroke();
                }
            }
        }
    }
    // 获取页面参数
    const queryParams = useQueryParams();
    const basicInfo: AnalysisPageProps = {
        projectName: queryParams['projectName'] || "未知项目",
        jsonPath: queryParams['jsonPath'] || "",
    };
    // 退出页面
    const quitPage = () => {
        navigateTo("/pages/demo/das");
    }
    // TODO: 保存更改
    const saveChanges = ()=>{
        console.log('TODO')
    }
    /*
     * 画布
     */
    // 处理拖拽
    const handleMouseDown = (e: React.MouseEvent) => {
        if (e.button === 0) { // 左键
            setIsDragging(true);
            setDragStart({ x: e.clientX, y: e.clientY });
        }
    };
    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging && dragStart) {
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            setOffset((prevOffset) => ({
                x: prevOffset.x + dx,
                y: prevOffset.y + dy,
            }));
            setDragStart({ x: e.clientX, y: e.clientY });
        }
    };
    const handleMouseUp = () => {
        setIsDragging(false);
    };
    const handleWheel = (e: React.WheelEvent) => {
        const scaleChange = e.deltaY < 0 ? 1.1 : 0.9; // 上滚放大，下滚缩小
        setScale((prevScale) => Math.max(0.01, Math.min(50, prevScale * scaleChange))); // 限制缩放范围
    };
    useEffect(() => {
        if (canvasRef.current) {
            initCanvas(projectJson); // 重绘线条
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
        }
    }, [offset, scale, projectJson]);



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
                    <Box className="flex-grow border-2 border-gray-400 mb-4 rounded-lg overflow-hidden">
                        <canvas ref={canvasRef}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
                                onWheel={handleWheel}
                                style={{width: "100%", height: "100%", background: "white"}} />
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
