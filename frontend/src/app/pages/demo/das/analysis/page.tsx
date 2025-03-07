"use client";

import React, {useEffect, useRef, useState} from "react";
import {useQueryParams} from "@/app/utils/navigator";
import { getJsonObj } from "@/app/components/draw/utils/das";
import CanvasComponent from "@/app/components/draw/CanvasComponent";
import TotalControlPanel from "@/app/pages/demo/das/analysis/coms/TotalControlPanel";
import ResizableDialog from "@/app/components/ResizableDialog";
import StdCom from "@/app/pages/demo/das/analysis/coms/panles/StdCom";
import CurCom from "@/app/pages/demo/das/analysis/coms/panles/CurCom";
import RelayCom from "@/app/pages/demo/das/analysis/coms/panles/RelayCom";

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
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [scale, setScale] = useState(1);

    // 展示Modal TODO all false
    const [showStdCom, setShowStd] = useState(false);
    const [showCurCom, setShowCurCom] = useState(true);
    const [showRelay, setShowRelay] = useState(false);

    const handleShowStdChange = () => {
        setShowStd(true);
        setShowCurCom(false);
        setShowRelay(false);
        setOpenPanel(true);
    }
    const handleShowCurComChange = () => {
        setShowCurCom(true);
        setShowStd(false);
        setShowRelay(false);
        setOpenPanel(true);
    }
    const handleShowRelayChange = () => {
        setShowRelay(true);
        setShowCurCom(false);
        setShowStd(false);
        setOpenPanel(true);
    }

    // 底部面板 TODO false
    const [openPanel, setOpenPanel] = useState(true);
    const closePanel = () => {
        setOpenPanel(false);
        setShowRelay(false);
        setShowCurCom(false);
        setShowStd(false);
    }

    // 获取页面参数
    const queryParams = useQueryParams();
    const basicInfo: AnalysisPageProps = useRef<AnalysisPageProps>({
        projectName: queryParams['projectName'] || "未知项目",
        jsonPath: queryParams['jsonPath'] || "",
    }).current;

    // 更改块的显示状态
    const changeBlockMarkShow = (blockHandle: number[]) => {
        const blocks = projectJson?.USED_BLOCKS
        if (blocks) {
            const block = blocks.find((item: any) => item.handle === blockHandle)
            if (block) {
                block.showMark = !block.showMark
            }
        }
        setProjectJson({...projectJson})
    }

    const changeAllBlockMarks = (current: boolean) => {
        const blocks = projectJson?.USED_BLOCKS
        if (blocks) {
            blocks.forEach((item: any) => {
                item.showMark = !current
            })
        }
        setProjectJson({...projectJson})
    }

    const canvasComRef = useRef(null);
    const handleCanvasFocus = (centerPt: any, maxBoxSize: any) => {
        const canvas = getComCanvas()
        if (centerPt && maxBoxSize && canvas) {
            const minCanvasSize = Math.min(canvas.width, canvas.height);
            const scale = minCanvasSize / (maxBoxSize * 10)
            console.log("focus", centerPt, maxBoxSize, scale)
            setOffset({ x: -1 *centerPt[0], y: -1 * centerPt[1] });
            setScale(scale);
        }
    }

    const getComCanvas = () => {
        return canvasComRef.current?.getCanvas()
    }

    useEffect(() => {
        console.log(basicInfo, '载入项目-基础信息');
        getJsonObj(basicInfo.jsonPath).then(res=>{
            console.log(res, '处理json数据');
            setProjectJson(res)
        })
    }, []);

    return (
        <div>
            <CanvasComponent
                ref={canvasComRef}
                projectJson={projectJson}
                offset={offset}
                scale={scale}
                onOffsetChange={setOffset}
                onScaleChange={setScale}
            />
            <TotalControlPanel
                projectName={basicInfo.projectName}
                handleShowCurComChange={handleShowCurComChange}
                handleShowRelayChange={handleShowRelayChange}
                handleShowStdChange={handleShowStdChange}
            />
            <ResizableDialog
                isOpen={openPanel}
                onClose={closePanel}
            >
                {showStdCom && <StdCom/>}
                {showCurCom && <CurCom
                    usedBlocks={projectJson?.USED_BLOCKS}
                    changeShowMark={changeBlockMarkShow}
                    changeAllShowMark={changeAllBlockMarks}
                    canvasFocus={handleCanvasFocus}
                />}
                {showRelay && <RelayCom/>}
            </ResizableDialog>
        </div>
    );
}
