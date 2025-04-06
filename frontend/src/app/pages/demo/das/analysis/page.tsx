"use client";

import React, {useCallback, useEffect, useRef, useState} from "react";
import {useQueryParams} from "@/app/utils/navigator";
import {getJsonObj} from "@/app/components/draw/utils/das";
import CanvasComponent from "@/app/components/draw/CanvasComponent";
import TotalControlPanel from "@/app/pages/demo/das/analysis/coms/TotalControlPanel";
import ResizableDialog from "@/app/components/ResizableDialog";
import StdCom from "@/app/pages/demo/das/analysis/coms/panles/StdCom";
import CurCom from "@/app/pages/demo/das/analysis/coms/panles/CurCom";
import RelayCom, {CytoscapeElement} from "@/app/pages/demo/das/analysis/coms/panles/RelayCom";
import SvgGetter from "@/app/components/draw/SvgGetter";
import {createRoot} from "react-dom/client";
import {err} from "@/app/utils/alerter";
import {analysisPublicProjectGraphML, getProjectComponents, getProjectGraph} from "@/app/api/das";
import {checkValidBlock} from "@/app/components/draw/utils/drawCalc";
import {MyResponse} from "@/app/types/common";
import {Box} from "@mui/material";

interface AnalysisPageProps {
    projectName: string;
    jsonPath: string;
    analysised: number;
    projectId: number;
}

interface CanvasComponent {
    getCanvas: () => HTMLCanvasElement
    getInserts: () => any[]
    getPipeline: () => any[]
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
    const [showRelay, setShowRelay] = useState(false);
    const [showStdCom, setShowStd] = useState(false);
    const [showCurCom, setShowCurCom] = useState(false);

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
        analysised: Number(queryParams['analysised']),
        projectId: Number(queryParams['projectId']),
    }).current;

    // 更改块的显示状态
    const changeBlockMarkShow = useCallback((blockHandle: number[]) => {
        setProjectJson((prev: any) => {
            if (!prev?.USED_BLOCKS) return prev;

            // 创建新数组和新对象
            const newBlocks = prev.USED_BLOCKS.map((block: any) =>
                (block.handle[1] === blockHandle[0] && block.handle[2] === blockHandle[1])
                    ? { ...block, showMark: !block.showMark } // 创建新对象
                    : block
            );

            return { ...prev, USED_BLOCKS: newBlocks }; // 返回新对象
        });
    }, [])

    const changeAllBlockMarks = useCallback((current: boolean) => {
        setProjectJson((prev: any) => {
            if (!prev?.USED_BLOCKS) return prev;

            // 创建新数组和新对象
            const newBlocks = prev.USED_BLOCKS.map((block: any) => ({
                ...block,
                showMark: !current // 创建新对象
            }));

            return { ...prev, USED_BLOCKS: newBlocks };
        });
    }, [])

    const canvasComRef = useRef<CanvasComponent>(null);
    const handleCanvasFocus = useCallback((centerPt: any, maxBoxSize: any) => {
        const canvas = getComCanvas()
        if (centerPt && maxBoxSize && canvas) {
            const minCanvasSize = Math.min(canvas.width, canvas.height);
            const newScale = minCanvasSize / (maxBoxSize * 10);
            const newOffset = { x: -1 * centerPt[0], y: -1 * centerPt[1] };

            let progress = 0;
            const duration = 500; // 过渡时间（ms）
            const startTime = performance.now();

            const animate = (time: number) => {
                progress = Math.min((time - startTime) / duration, 1);
                setOffset(prev => ({
                    x: prev.x + (newOffset.x - prev.x) * progress,
                    y: prev.y + (newOffset.y - prev.y) * progress
                }));
                setScale(prev => prev + (newScale - prev) * progress);

                if (progress < 1) requestAnimationFrame(animate);
            };

            requestAnimationFrame(animate);
            showFocusRound();
        }
    }, [])
    const [roundShow, setRoundShow] = useState(false);
    const timeout = useRef<any>();
    const showFocusRound = () => {
        setRoundShow(true);
        if (timeout.current) {
            clearTimeout(timeout.current);
        }
        timeout.current = setTimeout(() => {
            setRoundShow(false);
        }, 1000)
    }

    /**
     * 子组件方法
     */
    // 获取canvas dom
    const getComCanvas = () => {
        return canvasComRef.current?.getCanvas()
    }

    // 获取所有inserts
    const getInserts = () : any[] => {
        return canvasComRef.current?.getInserts() || []
    }

    const getPipeline = () : any[] => {
        return canvasComRef.current?.getPipeline() || []
    }

    /**
     * 二次解析
     */
    // 解析块，插入，管道
    const analysisBlockAndPipes = async (handledProjectJson: any) => {
        try {
            let postData: any = {
                projectId: basicInfo.projectId,
                blockData: null,
                insertsData: null,
                pipesData: null
            }
            // 获取所有块，填入postData，handle svg name count
            postData.blockData = await getAllBlocks(handledProjectJson)
            // 解析所有插入 存储中心位置、盒子、handle
            postData.insertsData = getInserts()
            // 解析所有管道 记录所有线段（拆分折线）
            postData.pipesData = getPipeline()
            console.log(postData, 'Graph解析')
            return analysisPublicProjectGraphML(postData)
        } catch (error) {
            err('解析失败, ' + error);
            return null;
        }
    };

    const getAllBlocks = async (handledProjectJson: any) => {
        const blocks = handledProjectJson.USED_BLOCKS || [];
        const svgDataPromises = blocks.map((block: any) => {
            return new Promise<string>((resolve) => {
                // 创建隐藏容器
                const hiddenDiv = document.createElement('div');
                hiddenDiv.style.display = 'none';
                document.body.appendChild(hiddenDiv);

                // 创建组件并挂载
                const root = createRoot(hiddenDiv);
                const comTypeProps = block.original_entities.TYPES;

                root.render(
                    <SvgGetter
                        comTypeProps={comTypeProps}
                        onSvgStringGenerated={(svgProps) => {
                            // @ts-ignore
                            resolve(svgProps);
                            setTimeout(() => {
                                root.unmount(); // 卸载组件
                                document.body.removeChild(hiddenDiv); // 移除隐藏容器
                            }, 0)
                        }}
                    />
                );
            });
        });
        const svgResults = await Promise.all(svgDataPromises);
        const blockData = blocks.map((block: any, index: number) => {
            if (!checkValidBlock(block)) {return null}
            return {
                handle0: block.handle[1],
                handle1: block.handle[2],
                name: block.name,
                count: block?.inserts?.length || 0,
                ...svgResults[index]
            }
        }).filter(Boolean);
        return blockData;
    }

    const handleAnalysisFail = (e: any) => {
        console.error(e);
        err('解析失败');
        setLoading(false);
    }
    /**
     * 获取解析结果
     */
    // 获取解析结果
    const getAnalysisResults = () => {
        Promise.all([getAnalysisResultComponents(), getAnalysisResultGraph()]).then(()=>{
            console.log('获取全部解析结果成功')
            setLoading(false);
        }).catch(e => {
            handleAnalysisFail(e)
        })
    }
    // 获取图解析结果
    const [currentGraph, setCurrentGraph] = useState<CytoscapeElement[]>([])
    const any2CytoscapeElement = (data: any) => {
        const { id, label, type, position, source, target, node, box, stream } = data
        return {
            data: {
                id,
                label,
                type,
                source,
                target,
            },
            position,
            node,
            box,
            stream
        }
    }
    const getAnalysisResultGraph = () => {
        const getRes = getProjectGraph(basicInfo.projectId)
        return getRes.then((res: MyResponse) => {
            if (res.success) {
                const filteredData = JSON.parse(
                    JSON.stringify(
                        res.data.map(any2CytoscapeElement
                        ), (key, value) => {
                    return value === null ? undefined : value;
                }));
                console.log(filteredData, '获取图解析结果')
                setCurrentGraph(filteredData)
                return
            }
            throw new Error('获取图结构解析结果失败');
        }).catch((e:any)=>{
            throw e
        })
    }
    // 获取块解析结果
    const [currentBlocks, setCurrentBlocks] = useState<any[]>([]);
    const getAnalysisResultComponents = () => {
        const getRes = getProjectComponents(basicInfo.projectId)
        return getRes.then((res:MyResponse) => {
            console.log(res, '获取组件解析结果')
            if (res?.success) {
                setCurrentBlocks(res.data)
                return
            }
            throw new Error('获取组件解析结果失败');
        }).catch(e => {
            throw e;
        })
    }


    /**
     * 全局
     */
    const [loading, setLoading] = useState(true);
    // 页面初始化
    useEffect(() => {
        console.log(basicInfo, '进入analysis页面，载入项目-基础信息');
        getJsonObj(basicInfo.jsonPath).then(res=>{
            console.log(res, '处理json数据结果');
            setProjectJson(res);
        })
    }, []);
    // 首次绘制完成
    const firstDrawComplete = async () => {
        console.log('firstDrawComplete');
        try {
            if (basicInfo.analysised === 0) {
                console.log(projectJson, '未分析，开始分析')
                // 提取出块和初筛管道等传递给后端处理
                const postRes = await analysisBlockAndPipes(projectJson)
                if (postRes?.success) {
                    getAnalysisResults()
                    return
                }
                handleAnalysisFail(postRes?.message)
            } else {
                console.log('已分析，直接展示')
                getAnalysisResults()
            }
        } catch (e) {
            handleAnalysisFail(e)
        }
    }

    return (
        <Box>
            <CanvasComponent
                ref={canvasComRef}
                projectJson={projectJson}
                offset={offset}
                scale={scale}
                onOffsetChange={setOffset}
                onScaleChange={setScale}
                onFirstDrawComplete={firstDrawComplete}
            />
            {
                roundShow && (
                    <Box className={`w-32 h-32 absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 rounded-full border-red-500 border-4 animate-pulse`}></Box>
                )
            }
            <TotalControlPanel
                projectName={basicInfo.projectName}
                handleShowCurComChange={handleShowCurComChange}
                handleShowRelayChange={handleShowRelayChange}
                handleShowStdChange={handleShowStdChange}
            />
            <ResizableDialog
                loading={loading}
                isOpen={openPanel}
                onClose={closePanel}
            >
                {showStdCom && <StdCom/>}
                {showCurCom && <CurCom
                    usedBlocks={currentBlocks}
                    changeShowMark={changeBlockMarkShow}
                    changeAllShowMark={changeAllBlockMarks}
                    canvasFocus={handleCanvasFocus}
                />}
                {showRelay && <RelayCom
                    sourceElements={currentGraph}
                    canvasFocus={handleCanvasFocus}
                />}
            </ResizableDialog>
        </Box>
    );
}
