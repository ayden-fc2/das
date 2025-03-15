// src/app/components/CanvasComponent.tsx
"use client";

import React, {useRef, useEffect, useState, forwardRef, useImperativeHandle} from "react";
import { Box } from "@mui/material";
import {
    drawArc,
    drawCircle,
    drawLine,
    drawLwpolyLine,
    drawMtext,
    drawText, handleInsert
} from "@/app/components/draw/utils/draw";
import {calcComBox} from "@/app/components/draw/utils/drawCalc";

interface CanvasComponentProps {
    projectJson: any;
    offset: { x: number; y: number };
    scale: number;
    onOffsetChange: (offset: { x: number; y: number }) => void;
    onScaleChange: (scale: number) => void;
    onFirstDrawComplete?: () => void;
}


const CanvasComponent = forwardRef(({
                                            projectJson,
                                            offset,
                                            scale,
                                            onOffsetChange,
                                            onScaleChange,
                                            onFirstDrawComplete,
                                            }: CanvasComponentProps, ref: any) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const isDragging = useRef(false);
    const dragStart = useRef<{ x: number; y: number } | null>(null);

    /**
     * 对外暴露方法
     * @param e
     */
    useImperativeHandle(ref, ()=>({
        getCanvas: () => canvasRef.current,
        getInserts: () => insertsRef.current,
        getPipeline: () => tilingRef.current,
    }))

    // 事件处理函数
    const handleMouseDown = (e: React.MouseEvent) => {
        if (e.button === 0) {
            isDragging.current = true;
            dragStart.current = { x: e.clientX, y: e.clientY };
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging.current && dragStart.current) {
            const deltaX = (e.clientX - dragStart.current.x) / scale;
            const deltaY = (dragStart.current.y - e.clientY) / scale;

            onOffsetChange({
                x: offset.x + deltaX,
                y: offset.y + deltaY,
            });
            dragStart.current = { x: e.clientX, y: e.clientY };
        }
    };

    const handleMouseUp = () => {
        isDragging.current = false;
    };

    const handleWheel = (e: React.WheelEvent) => {
        const scaleChange = e.deltaY < 0 ? 1.1 : 0.9;
        const newScale = Math.max(0.0001, Math.min(50, scale * scaleChange));
        onScaleChange(newScale);
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

    const tilingRef = useRef<any[]>([]);

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
        drawText(ctx, projectJson?.TYPES?.TEXT, scale)
        // 记录预选管道
        if (firstDraw) {
            for (const lineElement of projectJson?.TYPES?.LINE) {
                tilingRef.current.push({
                    startX: lineElement.start[0],
                    startY: lineElement.start[1],
                    endX: lineElement.end[0],
                    endY: lineElement.end[1],
                });
            }
            for (const lwpolylineElement of projectJson?.TYPES?.LWPOLYLINE) {
                for (let i = 0; i < lwpolylineElement.points.length - 1; i++) {
                    tilingRef.current.push({
                        startX: lwpolylineElement.points[i][0],
                        startY: lwpolylineElement.points[i][1],
                        endX: lwpolylineElement.points[i + 1][0],
                        endY: lwpolylineElement.points[i + 1][1],
                    })
                }
            }
        }
    }

    const insertsRef = useRef<any[]>([]);
    const [firstDraw, setFirstDraw] = useState<boolean>(true);

    // 绘制插入图形
    const drawInserts = (ctx: CanvasRenderingContext2D, projectJson: any, inserts: any, currentInsertDetail: any, depth: number = 0) => {
        // console.log('深度' + depth, inserts, currentInsertDetail);
        for (let i = 0; i < inserts.length; i++) {
            const blockEntities = projectJson.USED_BLOCKS[inserts[i].blockIndex].original_entities;
            let insertDetail = projectJson.USED_BLOCKS[inserts[i].blockIndex].inserts[inserts[i].insertIndex]
            insertDetail.ins_pt = [insertDetail.ins_pt[0] + currentInsertDetail.ins_pt[0], insertDetail.ins_pt[1] + currentInsertDetail.ins_pt[1]]
            insertDetail.rotation = (insertDetail.rotation + currentInsertDetail.rotation) % 360
            if (!Array.isArray(insertDetail.scale)) {
                insertDetail.scale = [insertDetail.scale, insertDetail.scale]
            }
            insertDetail.scale[0] = insertDetail.scale[0] * currentInsertDetail.scale[0]
            insertDetail.scale[1] = insertDetail.scale[1] * currentInsertDetail.scale[1]

            drawLine(ctx, blockEntities.TYPES.LINE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawLwpolyLine(ctx, blockEntities.TYPES.LWPOLYLINE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawCircle(ctx, blockEntities.TYPES.CIRCLE, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawArc(ctx, blockEntities.TYPES.ARC, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawMtext(ctx, blockEntities.TYPES.MTEXT, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            drawText(ctx, blockEntities.TYPES.TEXT, scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)

            const insertBox = calcComBox(blockEntities.TYPES)
            const originalCenterPt = [(insertBox.minX + insertBox.maxX) / 2, (insertBox.minY + insertBox.maxY) / 2]
            insertDetail.center_pt = handleInsert(originalCenterPt[0], originalCenterPt[1], insertDetail.scale, insertDetail.rotation, insertDetail.ins_pt)
            insertDetail.maxBoxSize = Math.max((insertBox.maxX - insertBox.minX), (insertBox.maxY - insertBox.minY))
            // 插入标注
            const invalid = insertBox.maxY === 0 && insertBox.minY === 0 && insertBox.maxX === 0 && insertBox.minX === 0
            if (projectJson.USED_BLOCKS[inserts[i].blockIndex].showMark && !invalid) {
                drawText(ctx, [
                    {
                        text_value: `${insertDetail.handle[2]}`,
                        ins_pt: [
                            insertBox.maxX,
                            insertBox.minY,
                        ],
                        height: Math.min((insertBox.maxY - insertBox.minY) / 2, insertBox.maxX - insertBox.minX),
                        color: {
                            rgb: projectJson.USED_BLOCKS[inserts[i].blockIndex].markColor
                        }
                    }
                ], scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale, true)
                drawLwpolyLine(ctx, [
                    {
                        points: [
                            [insertBox.minX - 1, insertBox.minY - 1],
                            [insertBox.minX - 1, insertBox.maxY + 1],
                            [insertBox.maxX + 1, insertBox.maxY + 1],
                            [insertBox.maxX + 1, insertBox.minY - 1],
                        ],
                        color: {
                            rgb: projectJson.USED_BLOCKS[inserts[i].blockIndex].markColor
                        },
                        flag: 1
                    }
                ], scale, insertDetail.ins_pt, insertDetail.rotation, insertDetail.scale)
            }

            if (firstDraw) {
                // 汇总inserts
                const centerPtX: number = (insertBox.minX + insertBox.maxX) / 2
                const centerPtY: number = (insertBox.minY + insertBox.maxY) / 2
                const [handledX, handledY] = handleInsert(centerPtX, centerPtY, insertDetail.scale, insertDetail.rotation, insertDetail.ins_pt)
                const boxWidth = insertBox.maxX - insertBox.minX
                const boxHeight = insertBox.maxY - insertBox.minY
                if (boxWidth > 0 && boxHeight > 0) {
                    insertsRef.current.push({
                        handle0: insertDetail.handle[1],
                        handle1: insertDetail.handle[2],
                        centerPt: [handledX, handledY],
                        boxWidth: boxWidth,
                        boxHeight: boxHeight,
                        blockHandle0: projectJson.USED_BLOCKS[inserts[i].blockIndex].handle[1],
                        blockHandle1: projectJson.USED_BLOCKS[inserts[i].blockIndex].handle[2],
                    })
                }
            }

            // 递归insert
            const nextInserts = blockEntities.filter((item: any) => {
                return item.$ref
            })
            // console.log(nextInserts, '下一层')
            for (let i = 0; i < nextInserts.length; i++) {
                const refPath = nextInserts[i].$ref.split('.')
                const blockIndex = parseInt(refPath[1].match(/\d+/)[0]);  // 获取 USED_BLOCKS[0] 中的 0
                const insertIndex = parseInt(refPath[2].match(/\d+/)[0]); // 获取 inserts[0] 中的 0
                nextInserts[i].blockIndex = blockIndex;
                nextInserts[i].insertIndex = insertIndex;
            }
            if (nextInserts.length > 0) {
                drawInserts(ctx, projectJson, nextInserts, insertDetail, depth + 1)
            }
        }
    }

    // 初始化画布
    const initCanvas = () => {
        const canvas = canvasRef.current;
        if (!canvas || !projectJson) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width;
        canvas.height = height;

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        // 坐标系变换
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.scale(scale, -scale);
        ctx.translate(offset.x, offset.y);

        // 绘制内容
        drawAxis(ctx, canvas);
        drawTiling(ctx, projectJson);
        drawInserts(ctx, projectJson, projectJson?.TYPES?.INSERTS ?? [], {
            ins_pt: [0, 0],
            rotation: 0,
            scale: [1, 1, 1]
        });
        if (firstDraw) {
            setFirstDraw(false)
            if (onFirstDrawComplete) {
                onFirstDrawComplete(); // 触发回调
            }
        }
    };

    useEffect(() => {
        initCanvas();
    }, [offset, scale, projectJson]);

    return (
        <Box
            sx={{
                height: "100vh",
                position: "relative",
                overflow: "hidden"
            }}
        >
            <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                style={{ width: "100%", height: "100%", background: "white" }}
            />
        </Box>
    );
})

export default CanvasComponent;