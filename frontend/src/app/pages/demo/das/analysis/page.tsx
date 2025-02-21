"use client";

import React, {useEffect, useRef, useState} from "react";
import {navigateTo, useQueryParams} from "@/app/utils/navigator";
import { Button, Box, Typography } from "@mui/material"; // 引入Material UI组件
import { getJsonObj } from "@/app/utils/das";
import {drawArc, drawCircle, drawLine, drawLwpolyLine, drawMtext} from "@/app/pages/demo/das/analysis/utils/draw";



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
            console.log(res);
            setProjectJson(res)
        })
    }, []);

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
            const deltaX = (e.clientX - dragStart.x) / scale;
            const deltaY = (dragStart.y - e.clientY) / scale; // Y方向取反

            setOffset((prevOffset) => ({
                x: prevOffset.x + deltaX,
                y: prevOffset.y + deltaY, // 注意这里是加法
            }));
            setDragStart({ x: e.clientX, y: e.clientY });
        }
    };
    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleWheel = (e: React.WheelEvent) => {
        const scaleChange = e.deltaY < 0 ? 1.1 : 0.9; // 上滚放大，下滚缩小
        setScale((prevScale) => Math.max(0.0001, Math.min(50, prevScale * scaleChange))); // 限制缩放范围
    };

    // 绘制坐标轴
    const drawAxis = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
        const viewportWidth = (canvas.width / scale);
        const viewportHeight = (canvas.height / scale);

        const visibleLeft = -offset.x - viewportWidth / 2;
        const visibleRight = -offset.x + viewportWidth / 2;
        const visibleBottom = -offset.y - viewportHeight / 2; // Y轴已翻转
        const visibleTop = -offset.y + viewportHeight / 2;

        // X轴（红色）
        ctx.beginPath();
        ctx.moveTo(visibleLeft, 0);
        ctx.lineTo(visibleRight, 0);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 0.5 / scale;
        ctx.stroke();

        // Y轴（蓝色）
        ctx.beginPath();
        ctx.moveTo(0, visibleBottom);
        ctx.lineTo(0, visibleTop); // 注意方向已翻转
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 0.5 / scale;
        ctx.stroke();
    }

    // 绘制平铺图形
    const drawTiling = (ctx: CanvasRenderingContext2D, projectJson: any) => {
        // 绘制线段
        drawLine(ctx, projectJson?.TYPES?.LINE, scale)
        // 绘制折线
        drawLwpolyLine(ctx, projectJson?.TYPES?.LWPOLYLINE, scale)
        // 绘制圆形
        drawCircle(ctx, projectJson?.TYPES?.CIRCLE, scale)
        // 绘制曲线
        drawArc(ctx, projectJson?.TYPES?.ARC, scale)
        // 绘制文字
        drawMtext(ctx, projectJson?.TYPES?.MTEXT, scale)
    }

    // 绘制插入图形
    const drawInserts = (ctx: CanvasRenderingContext2D, projectJson: any) => {
        const inserts = projectJson?.TYPES?.INSERTS ?? []
        for (let i = 0; i < inserts.length; i++) {
            const blockEntities = projectJson.USED_BLOCKS[inserts[i].blockIndex].original_entities;
            const insertDetail = projectJson.USED_BLOCKS[inserts[i].blockIndex].inserts[inserts[i].insertIndex]
            drawLine(ctx, blockEntities.TYPES.LINE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawLwpolyLine(ctx, blockEntities.TYPES.LWPOLYLINE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawCircle(ctx, blockEntities.TYPES.CIRCLE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawArc(ctx, blockEntities.TYPES.ARC, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawMtext(ctx, blockEntities.TYPES.MTEXT, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
        }
    }

    // 渲染画布
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

                // 坐标轴修正
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.translate(canvas.width / 2, canvas.height / 2);
                ctx.scale(scale, -scale);
                ctx.translate(offset.x, offset.y);

                // 绘制坐标轴
                drawAxis(ctx, canvas);

                // 绘制平面图形
                drawTiling(ctx, projectJson);

                // 绘制插入图形
                drawInserts(ctx, projectJson)
            }
        }
    }

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
