"use client"

import {Box} from "@mui/material";
import React, {useEffect, useRef, useState} from "react";
import CytoscapeComponent from 'react-cytoscapejs';

/**
 * 渲染图结构，支持选中关键节点，展示它的基本信息和上下游关系
 * TODO: 使用GNN实现故障排查
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
    box?: any
}

export interface RelayComProps {
    sourceElements: CytoscapeElement[]
}

const RelayCom = React.memo(({sourceElements}: RelayComProps)=> {
    /**
     * 图数据
     */
    // 图结构数据
    const [elements, setElements] = useState<CytoscapeElement[]>([]);

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

    // 节点和连线样式
    const stylesheet = [
        ...elements
            .filter(element => element.data.type === "key") // 过滤出节点
            .map(element => ({
                selector: `node[id='${element.data.id}']`, // 根据 id 选择节点
                style: {
                    "background-color": getNodeColorByLabel(element.data.label), // 使用 data.color
                    "shape": "ellipse", // 节点形状为圆形
                    "label": element.data.label + ' ' + element.data.id, // 显示 label
                    "width": Math.min(element.box.width, element.box.height),
                    "height": Math.min(element.box.height, element.box.width),
                    "font-size": (Math.min(element.box.width, element.box.height) / 2),
                },
            })),
        {
            selector: "node[type='virtual']", // 类型2的节点
            style: {
                "background-color": "#666",
                "shape": "ellipse",
                "label": "data(label)",
                "width": 10,
                "height": 10,
            },
        },
        {
            selector: "edge", // 边
            style: {
                "width": 2,
                "line-color": "#6b9aff88",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
            },
        },
    ];

    // 节点点击事件
    const handleNodeClick = (e: any) => {
        console.log(e)
    }

    /**
     * 全局方法
     */
    useEffect(() => {
        initElementsColor(sourceElements)
        setElements(sourceElements);
    }, [sourceElements])

    return (
        <Box className={`w-full h-full text-black overscroll-hidden flex`}>
            <Box className={`h-full flex-1 mr-2 relative rounded-md bg-gray-100`}>
                {elements.length && (
                    <CytoscapeComponent
                        elements={elements}
                        stylesheet={stylesheet}
                        style={ { width: '100%', height: '100%' } }
                        layout={{
                            name: 'preset',
                            fit: true
                        }}
                        cy={(cy: any) => {
                            cy.on('click', 'node', (e: any) => {
                                handleNodeClick(e)
                            })
                        }}
                    />
                )}
            </Box>
            <Box className={`flex-1 ml-2`}>

            </Box>
        </Box>
    )
})

export default RelayCom