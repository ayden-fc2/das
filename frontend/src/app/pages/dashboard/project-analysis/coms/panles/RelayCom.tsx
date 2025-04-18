"use client"

import {
    Box,
    Button,
    ButtonGroup,
    Paper,
    Switch,
    Table, TableBody, TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography
} from "@mui/material";
import React, {useEffect, useRef, useState} from "react";
import CytoscapeComponent from 'react-cytoscapejs';
import SvgRender from "@/app/components/draw/SvgRender";
import {err, info} from "@/app/utils/alerter";
import {genTraceApi} from "@/app/api/das";

/**
 * 渲染图结构，支持选中关键节点，展示它的基本信息和上下游关系
 * @constructor
 */
export interface CytoscapeElement {
    data: {
        id?: string;
        label?: string;
        type?: "key" | "virtual"; // 节点类型标识
        source?: string;
        target?: string;
    };
    position?: {
        x: number;
        y: number;
    };
    node?: any,
    box?: any,
    stream?: any,
}

export interface RelayComProps {
    sourceElements: CytoscapeElement[],
    canvasFocus: (centerPt: any, maxBoxSize: any) => void,
    projectId: number
}

const RelayCom = React.memo(({sourceElements, canvasFocus, projectId}: RelayComProps)=> {
    /**
     * 图数据
     */
    // 图结构数据
    const [elements, setElements] = useState<CytoscapeElement[]>([]);
    const transformedElements = elements.map(element => ({
        ...element,
        position: element.position
            ? { x: element.position.x, y: -1 * element.position.y }
            : undefined
    }))

    /**
     * 样式
     */

    const colors = useRef<Map<string, string>>(new Map());
    // 初始化随机颜色
    var randomColor = require('randomcolor')
    const initElementsColor = (elements: CytoscapeElement[]) => {
        for(let i = 0; i < elements.length; i++) {
            const label = elements[i].data.label;
            if (label && !colors.current.has(label)) {
                const color =randomColor(
                    {
                        luminosity: 'dark',
                        hue: 'random',
                    }
                )
                colors.current.set(label, color)
            }
        }
    }
    // 获取颜色
    const getNodeColorByLabel = (label: string | undefined) => {
        if (!label) {
            return "#666666"
        }
        return colors.current.get(label);
    }
    // 初始化图形尺寸
    const errorBase = useRef<number>(200);
    const initErrorBase = (elements: CytoscapeElement[]) => {
        let eb = 0
        let boxNumber = 0
        for(let i = 0; i < elements.length; i++) {
            if (elements[i]?.box) {
                eb += elements[i].box.width + elements[i].box.height
                boxNumber++
            }
        }
        eb = eb / boxNumber
        errorBase.current = eb
    }

    // 节点和连线样式
    const stylesheet = [
        ...elements
            .filter(element => element.data.type === "key") // 过滤出节点
            .map(element => ({
                selector: `node[id='${element.data.id}']`, // 根据 id 选择节点
                style: {
                    "background-color": getNodeColorByLabel(element.data.label), // 使用 data.color
                    // "label": element.data.label + ' ' + element.data.id, // 显示 label
                    "shape": "ellipse", // 节点形状为圆形
                    "width": Math.min(element.box.width, element.box.height),
                    "height": Math.min(element.box.height, element.box.width),
                    // "shape": "round-rectangle", // 节点形状为圆形
                    // "width": element.box.width,
                    // "height": element.box.height,
                    "font-size": (Math.min(element.box.width, element.box.height) / 2),
                },
            })),
        {
            selector: "node[type='virtual']", // 类型2的节点
            style: {
                "background-color": "#666",
                "shape": "ellipse",
                "label": "data(label)",
                "width": errorBase.current * 0.05,
                "height": errorBase.current * 0.05,
            },
        },
        {
            selector: "edge", // 边
            style: {
                "width": errorBase.current * 0.01,
                "line-color": "#6b9aff88",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": errorBase.current * 0.01
            },
        },
        {
            selector: 'node.focused', // 选中节点样式
            style: {
                'border-width': errorBase.current * 0.02,
                'border-color': '#f3f4f6',
                'outline-width': errorBase.current * 0.02,
                'outline-color': 'green',
            },
        },
        {
            selector: 'node.fault', // 选中节点样式
            style: {
                'label': 'F', // 显示字母F
                'color': '#fff', // 文字颜色
                'text-valign': 'center', // 垂直居中
                'text-halign': 'center', // 水平居中
                'text-outline-width': 1, // 文字描边
                'z-index': 999, // 确保在最上层
                'font-size': 'auto', // 根据节点尺寸自动调整
                'text-outline-color': 'blue', // 白色描边保证对比度
                'border-width': errorBase.current * 0.02,
                'border-color': '#f3f4f6',
                'outline-width': errorBase.current * 0.04,
                'outline-color': 'blue',
            },
        },
        {
            selector: 'node.reason',
            style: {
                'label': 'R', // 显示字母F
                'color': '#fff', // 文字颜色
                'text-valign': 'center', // 垂直居中
                'text-halign': 'center', // 水平居中
                'text-outline-width': 1, // 文字描边
                'z-index': 999, // 确保在最上层
                'font-size': 'auto', // 根据节点尺寸自动调整
                'text-outline-color': 'red', // 白色描边保证对比度
                'border-width': errorBase.current * 0.02,
                'border-color': '#f3f4f6',
                'outline-width': errorBase.current * 0.04,
                'outline-color': 'red', // 使用紫色边框标识原因节点
            },
        },
    ];

    /**
     * utils
     */
    // 获取某节点所有下游的节点
    function getNodeDownstreamRelations(focusedNode: CytoscapeElement | undefined, sourceElements: CytoscapeElement[]) {
        let result: any[] = []
        const downStreamNodeIdsString: string = focusedNode?.stream?.downstream || ""
        const downStreamNodeIds = downStreamNodeIdsString.split(",")
        for (let i = 0; i < downStreamNodeIds.length; i++) {
            for (let j = 0; j < sourceElements.length; j++) {
                if (sourceElements[j].data.id === downStreamNodeIds[i]) {
                    result.push(
                        {
                            label: sourceElements[j].data.label || "",
                            id: sourceElements[j].data.id || "",
                            box: sourceElements[j].box,
                            position: sourceElements[j].position
                        }
                    )
                }
            }
        }
        return result
    }

    function getNodeUptreamRelations(focusedNode: CytoscapeElement | undefined, sourceElements: CytoscapeElement[]) {
        let result: any[] = []
        const upStreamNodeIdsString: string = focusedNode?.stream?.upstream || ""
        const upStreamNodeIds = upStreamNodeIdsString.split(",")
        for (let i = 0; i < upStreamNodeIds.length; i++) {
            for (let j = 0; j < sourceElements.length; j++) {
                if (sourceElements[j].data.id === upStreamNodeIds[i]) {
                    result.push(
                        {
                            label: sourceElements[j].data.label || "",
                            id: sourceElements[j].data.id || "",
                            box: sourceElements[j].box,
                            position: sourceElements[j].position
                        }
                    )
                }
            }
        }
        return result
    }

    const focusInnerClick = (centerPt: any, maxBoxSize: number) => {
        canvasFocus(centerPt, maxBoxSize)
    }

    const handleNodeFocus = (nodeId: any) => {
        const nodeToFocus = sourceElements.filter(item => item.data.id === nodeId)[0]
        focusInnerClick([nodeToFocus?.position?.x ?? 0, nodeToFocus?.position?.y ?? 0], Math.max(nodeToFocus.box.width, nodeToFocus.box.height))
    }

    /**
     * 节点点击事件
     */
    const currentTarget = useRef<any>();
    const [focusedNode, setFocusedNode] = useState<CytoscapeElement>();
    const handleNodeClick = (e: any) => {
        currentTarget.current = e.target
        // 设置当前选中节点
        const currentId = e.target.data('id')
        const currentNode = sourceElements.find(node => node.data.id === currentId);
        setFocusedNode(currentNode)
        // 故障分析
        if (currentMode === 'fault') {
            if (faultMode === "add") {
                addFaultNode(currentNode, e)
            } else if (faultMode === "delete") {
                deleteFaultNode(currentNode, e)
            }
        }
    }

    /**
     * 全局方法
     */
    const [currentMode, setCurrentMode] = useState("info");
    useEffect(() => {
        initElementsColor(sourceElements)
        initErrorBase(sourceElements)
        setElements(sourceElements);
    }, [sourceElements])

    /**
     * 故障排查
     */
    const [faultMode, setFaultMode] = useState("add");
    const [faultNodes, setFaultNodes] = useState<any[]>([]);

    // 添加节点
    const addFaultNode = (node: CytoscapeElement | undefined, e: any) => {
        e.target.removeClass('fault');
        e.target.addClass('fault');
        const id = node?.data?.id;
        if (id !== undefined) {
            setFaultNodes(prev =>
                prev.includes(id) ? prev : [...prev, id]
            );
        }
    };

    // 删除节点
    const deleteFaultNode = (node: CytoscapeElement | undefined, e: any) => {
        const id = node?.data?.id;
        if (id !== undefined) {
            setFaultNodes(prev => prev.filter(item => item !== id));
        }
        e.target.removeClass('fault');
    };

    const [predicts, setPredicts] = useState<any[]>([]);
    const e = useRef<any>();
    const runAnalysis = () => {
        if (faultNodes.length === 0) {
            err('Please select at least one fault node to run analysis.')
            return
        }
        const getResult = genTraceApi(projectId, faultNodes.join(","))
        getResult.then(res => {
            e.current.nodes().removeClass('reason');
            setPredicts(res.data.predictions);
            const topReasonNodeIds = res.data.predictions
                .slice(0, 3) // 取置信度前三的节点
                .map((p: any) => p.node_id);

            topReasonNodeIds.forEach((id: string) => {
                const node = e.current.getElementById(id);
                if (node) {
                    node.addClass('reason');
                }
            });
        }).catch(e => {
            err('something went wrong!')
        })
    }

    return (
        <Box className={`w-full h-full text-black overscroll-hidden flex`}>
            <Box className={`h-full flex-[.8] mr-2 relative rounded-md bg-gray-100`}>
                {elements.length && (
                    <CytoscapeComponent
                        elements={transformedElements}
                        stylesheet={stylesheet}
                        style={ { width: '100%', height: '100%' } }
                        layout={{
                            name: 'preset',
                            fit: true
                        }}
                        cy={(cy: any) => {
                            e.current = cy
                            cy.on('click', 'node', (e: any) => {
                                // 只有点击关键节点时有反应
                                if (e.target.data().type === 'key') {
                                    // 处理focus
                                    cy.nodes().removeClass('focused');
                                    e.target.addClass('focused');
                                    handleNodeClick(e)
                                }
                            })
                        }}
                    />
                )}
            </Box>
            <Box className={`flex-1 ml-2 flex flex-col`}>
                {/*选择模式*/}
                <ButtonGroup size={"small"} className={"w-full flex align-center items-center justify-center mb-2"}>
                    <Button
                        disabled={currentMode === "info"}
                        onClick={() => {setCurrentMode("info")}}
                    >
                        Basic Info
                    </Button>
                    <Button
                        disabled={currentMode === "fault"}
                        onClick={() => {setCurrentMode("fault")}}
                    >
                        Fault Analysis
                    </Button>
                </ButtonGroup>
                {currentMode === "info" && (<Box className={`flex-1 w-full flex flex-col relative overscroll-y-auto overflow-x-hidden`}>
                    {/*提示*/}
                    {!focusedNode &&
                        (<Box className={`w-full h-full flex align-center items-center justify-center mb-2`}>
                            <Typography variant="body2">
                                Please select a key node to show its basic info and upstream/downstream relations.
                            </Typography>
                        </Box>)
                    }
                    {/*块属性*/}
                    {focusedNode && (
                        <Box className={`w-full h-36 flex relative border-b border-gray-200`}>
                            <Box className={`flex-1 h-full flex flex-col relative justify-center`}>
                                <Box className={`flex justify-between items-center px-2 mb-2`}>
                                    <Typography variant="body2" fontWeight={"bold"}>Basic Info</Typography>
                                    <Button size={"small"} onClick={()=>{focusInnerClick([focusedNode?.position?.x, focusedNode?.position?.y], Math.max(focusedNode?.box.width, focusedNode?.box.height))}}>Focus</Button>
                                </Box>
                                <TableContainer component={Paper}>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell align="center"><strong>Block</strong></TableCell>
                                                <TableCell align="center"><strong>Id</strong></TableCell>
                                                <TableCell align="center"><strong>Position (x * y)</strong></TableCell>
                                                <TableCell align="center"><strong>Size (w * h)</strong></TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            <TableRow>
                                                <TableCell align="center">
                                                    <strong>{focusedNode?.data.label}</strong>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <strong>{focusedNode?.data.id}</strong>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <strong>
                                                        {focusedNode?.position?.x.toFixed(2)} * {focusedNode?.position?.y.toFixed(2)}
                                                    </strong>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <strong>
                                                        {focusedNode?.box.width.toFixed(2)} * {focusedNode?.box.height.toFixed(2)}
                                                    </strong>
                                                </TableCell>
                                            </TableRow>
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Box>
                            <Box className={`h-32 w-36 p-4 mx-4 bg-gray-100 rounded-md flex flex-col justify-center items-center`}>
                                <SvgRender svgString={focusedNode?.node?.svg}/>
                            </Box>
                        </Box>
                    )}
                    {/*上下游关系*/}
                    {focusedNode && (
                        <Box className={`w-full flex justify-between relative mt-4`}>
                            <Box className={`flex-1 m-2`}>
                                <Typography variant="body2" fontWeight={"bold"}>Upstream Relations</Typography>
                                <TableContainer component={Paper} className={`mt-2`}>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell align="center">Index</TableCell>
                                                <TableCell align="center">Block</TableCell>
                                                <TableCell align="center">Id</TableCell>
                                                <TableCell align="center">Operation</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {getNodeUptreamRelations(focusedNode, sourceElements).map((relation, index) => (
                                                <TableRow key={index}>
                                                    <TableCell align="center">{index + 1}</TableCell>
                                                    <TableCell align="center">{relation.label}</TableCell>
                                                    <TableCell align="center">{relation.id}</TableCell>
                                                    <TableCell align="center">
                                                        <Button size={"small"} onClick={()=> {
                                                            focusInnerClick([relation.position.x, relation.position.y], Math.max(relation.box.width, relation.box.height))
                                                        }}>focus</Button>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Box>
                            <Box className={`flex-1 m-2`}>
                                <Typography variant="body2" fontWeight={"bold"}>Downstream Relations</Typography>
                                <TableContainer component={Paper} className={`mt-2`}>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell align="center">Index</TableCell>
                                                <TableCell align="center">Block</TableCell>
                                                <TableCell align="center">Id</TableCell>
                                                <TableCell align="center">Operation</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {getNodeDownstreamRelations(focusedNode, sourceElements).map((relation, index) => (
                                                <TableRow key={index}>
                                                    <TableCell align="center">{index + 1}</TableCell>
                                                    <TableCell align="center">{relation.label}</TableCell>
                                                    <TableCell align="center">{relation.id}</TableCell>
                                                    <TableCell align="center">
                                                        <Button size={"small"} onClick={()=> {
                                                            focusInnerClick([relation.position.x, relation.position.y], Math.max(relation.box.width, relation.box.height))
                                                        }}>focus</Button>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Box>
                        </Box>
                    )}
                </Box>)
                }
                {currentMode === "fault" && (<Box className={`flex-1 w-full flex flex-col relative overscroll-y-auto overflow-x-hidden`}>
                    {/*表格*/}
                    <Box className={`w-full flex-1 pb-4 overflow-x-hidden overscroll-y-auto`}>
                        {/* 故障节点标志 */}
                        <Box className={`w-full h-12 flex items-center justify-between px-4`}>
                            <Box className={`flex items-center justify-between`}>
                                <Typography variant="body2" fontWeight={"bold"}>
                                    Fault Nodes:
                                </Typography>
                                <Box className="ml-4 w-8 h-8 text-blue-500 text-xl font-bold rounded-full border-2 border-blue-500 flex justify-center text-center align-center" >
                                    F
                                </Box>
                            </Box>
                            <Box className={`flex items-center justify-between`}>
                                <Typography variant="body2" fontWeight={"bold"}>
                                    Top Reasons:
                                </Typography>
                                <Box className="ml-4 w-8 h-8 text-red-500 text-xl font-bold rounded-full border-2 border-red-500 flex justify-center text-center align-center" >
                                    R
                                </Box>
                            </Box>
                            <Typography variant="body2" fontWeight={"bold"}>
                                Faults: {faultNodes.length}
                            </Typography>
                        </Box>
                        {/* 故障溯源结果表格 */}
                        <TableContainer component={Paper}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell align="center">Index</TableCell>
                                        <TableCell align="center">Id</TableCell>
                                        <TableCell align="center">Confidence</TableCell>
                                        <TableCell align="center">Operation</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {predicts.map((node, index) => (
                                        <TableRow key={index}>
                                            <TableCell align="center">{index + 1}</TableCell>
                                            <TableCell align="center">{node.node_id}</TableCell>
                                            <TableCell align="center">{node.predicted_confidence}</TableCell>
                                            <TableCell align="center">
                                                <Button size={"small"} onClick={()=> {
                                                    handleNodeFocus(node.node_id)
                                                    // TODO: focus操作
                                                    // focusInnerClick([node.position.x, node.position.y], Math.max(node.box.width, node.box.height))
                                                }}>focus</Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                    {/*按钮*/}
                    <Box className={`w-full h-16 my-2 flex justify-between items-center rounded-md border-t-2 border-gray-200`}>
                        {faultMode === "delete" && (
                            <>
                                <Typography variant="body2">Click to delete fault nodes</Typography>
                                <Button onClick={()=>{setFaultMode("add")}} variant={"contained"} color={"success"}>Add</Button>
                            </>
                        )}
                        {faultMode === "add" && (
                            <>
                                <Typography variant="body2">Click node to add fault nodes</Typography>
                                <Button onClick={()=>{setFaultMode("delete")}} variant={"contained"} color={"error"}>Delete</Button>
                            </>
                        )}
                        <Button onClick={()=>{
                            e.current.nodes().removeClass('reason');
                            e.current.nodes().removeClass('fault');
                            setFaultNodes([])
                            setPredicts([])
                        }} variant={"contained"} color={"secondary"}>Clear</Button>
                        <Button onClick={runAnalysis} variant={"contained"} color={"primary"}>Analysis</Button>
                    </Box>
                </Box>)}
            </Box>
        </Box>
    )
})

export default RelayCom