"use client"

import {
    Box, Checkbox,
    Collapse,
    IconButton,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow, TextField, Typography
} from "@mui/material"
import React, {useEffect, useMemo, useRef, useState} from "react"
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";

import {calcComBox, checkValidBlock} from "@/app/components/draw/utils/drawCalc";
import SvgRender from "@/app/components/draw/SvgRender";

interface CurComProps {
    usedBlocks: any[],
    changeShowMark: (handle: number[]) => void,
    changeAllShowMark: (checked: boolean) => void,
    canvasFocus: (centerPt: any, maxBoxSize: any) => void,
}

const CurCom = React.memo(({usedBlocks, changeShowMark, changeAllShowMark, canvasFocus}: CurComProps)=> {

    const focusInnerClick = (centerPt: any, maxBoxSize: number) => {
        canvasFocus(centerPt, maxBoxSize)
    }

    const getDynamicScale = (block: any) => {
        // 获取实体包围盒尺寸
        const entities = block.original_entities.TYPES;
        const { minX, maxX, minY, maxY } = calcComBox(entities);
        const contentWidth = maxX - minX;
        const contentHeight = maxY - minY;

        // 容器可用尺寸（144x128 对应 tailwind 的 w-36 h-32）
        const containerWidth = 144;
        const containerHeight = 128;

        // 计算缩放比例（保留5%边距）
        const widthRatio = (containerWidth * 0.95) / contentWidth;
        const heightRatio = (containerHeight * 0.95) / contentHeight;

        // 取最小比例确保完整显示
        return Math.min(widthRatio, heightRatio);
    };

    /**
     * 子组件管理
     */
    const [expandedBlocks, setExpandedBlocks] = useState<number[]>([])
    const [allChecked, setAllChecked] = useState<boolean | undefined>(false)
    const [checked, setChecked] = useState<Map<number, boolean>>(new Map())
    const toggleExpand = (blockHandle: number) => {
        let newArray = [...expandedBlocks];
        if (newArray.includes(blockHandle)) {
            newArray = newArray.filter((bh: number) => bh !== blockHandle);
        } else {
            newArray.push(blockHandle);
        }
        setExpandedBlocks(newArray);
    };
    const handleChangeAllShowMark = (checked: boolean) => {
        changeAllShowMark(checked);
        const target = !checked
        setAllChecked(target)
        setChecked(new Map(usedBlocks.map((block) => [block.handle1, target])))
    }
    const handleChangeShowMark = (handle0: number, handle1: number) => {
        changeShowMark([handle0, handle1])
        const newMap = new Map(usedBlocks.map((block) => {
            const currentChecked: boolean = checked.get(block.handle1) || false
            if (handle1 === block.handle1) {
                return [block.handle1, !currentChecked]
            }
            return [block.handle1, currentChecked]
        }))
        setChecked(newMap)
        let allCheckedValue = true
        newMap.forEach((block) => {
            if (!block) {
                allCheckedValue = false
            }
        })
        if (checked.size !== usedBlocks.length) {
            allCheckedValue = false
        }
        setAllChecked(allCheckedValue)
    }

    /**
     * 子组件
     */

    const Row = React.memo((props: { block: any; expanded: boolean; onToggle: () => void }) => {
        const { block, expanded, onToggle } = props;
        const [filterInput, setFilterInput] = React.useState('');
        const [filteredHandles, setFilteredHandles] = React.useState<number[] | null>(null);

        const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            const input = e.target.value;
            setFilterInput(input);
            // 解析输入为数字数组，过滤无效值
            const handles = input
                .split(' ')
                .map(str => parseInt(str.trim(), 10))
                .filter(num => !isNaN(num));
            setFilteredHandles(handles.length > 0 ? handles : null);
        };

        // 生成过滤后的插入项列表
        const filteredInserts = block.inserts
            .map((insert: any, originalIndex: number) => ({ insert, originalIndex }))
            .filter(({ insert }: any) =>
                {
                    return !filteredHandles || filteredHandles.includes(insert.handle1)
                }
            );

        //中心点
        const getCenterPt = (centerX: number, centerY: number) => {
            return `${centerX.toFixed(2)}, ${centerY.toFixed(2)}`;
        }

        return (
            <React.Fragment>
                <TableRow>
                    {/*收缩展开*/}
                    <TableCell>
                        <IconButton
                            aria-label="expand row"
                            size="small"
                            onClick={() => toggleExpand(block.handle1)}>
                            {expanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                        </IconButton>
                    </TableCell>
                    {/*  Name  */}
                    <TableCell align="center" style={{color: '#' + block.markColor}}>{block.blockName}</TableCell>
                    {/*  Preview  */}
                    <TableCell align="center">
                        <Box className={`w-full flex justify-center items-center`}>
                            <Box className={`w-28 h-24 overflow-hidden flex justify-center items-center relative`}>
                                <SvgRender svgString={block.svg} />
                            </Box>
                        </Box>
                    </TableCell>
                    {/*  Inserts Number  */}
                    <TableCell align="center">{block.inserts.length}</TableCell>
                    {/*  Type Inference  */}
                    <TableCell align="center">TODO</TableCell>
                    {/*  Show Tags  */}
                    <TableCell align="center">
                        <Checkbox checked={checked.get(block.handle1)} onChange={() => handleChangeShowMark(block.handle0, block.handle1)} />
                    </TableCell>
                </TableRow>
                {/*  展开内容  */}
                <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
                        <Collapse in={expanded} unmountOnExit>
                        {/*<Collapse in={true} unmountOnExit>*/}
                            <Box sx={{ margin: 1 }}>
                                <Box className={`w-full flex justify-center items-center relative m-4`}>
                                    <TextField
                                        className={`w-full`}
                                        size="small"
                                        id={`index-filter-${block.handle1}`}
                                        label="Please input block handles, separated by space"
                                        value={filterInput}
                                        onChange={handleFilterChange}/>
                                </Box>
                                {/*  Inserts Table  */}
                                <Table size="small" aria-label="purchases">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell align="center">Handle</TableCell>
                                            <TableCell align="center">Center Position</TableCell>
                                            <TableCell align="center">Upstream</TableCell>
                                            <TableCell align="center">Downstream</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {filteredInserts.map(({ insert, originalIndex }: {insert: any, originalIndex: any}) => (
                                            getCenterPt(insert.centerX, insert.centerY) &&
                                            <TableRow key={originalIndex}>
                                                <TableCell align="center">{insert.handle1}</TableCell>
                                                <TableCell onClick={() => focusInnerClick([insert.centerX, insert.centerY], Math.max(insert.boxWidth, insert.boxHeight))} align="center">{getCenterPt(insert.centerX, insert.centerY)}</TableCell>
                                                <TableCell align="center">TODO</TableCell>
                                                <TableCell align="center">TODO</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>
                        </Collapse>
                    </TableCell>
                </TableRow>
            </React.Fragment>
        )
    })

    return (
        <Box className={`w-full h-full text-black overscroll-y-auto overflow-x-hidden`}>
            <Box className={`h-7 mb-2 flex px-2 items-center justify-center `}>
                <h1 className="text-blue-500 font-bold text-xl">Current DCS Components</h1>
                <span className="ml-2 text-gray-500">( total: {usedBlocks?.length} )</span>
            </Box>

            <TableContainer component={Paper}>
                <Table aria-label="collapsible table">
                    <TableHead>
                        <TableRow>
                            <TableCell align="center" />
                            <TableCell align="center">Name</TableCell>
                            <TableCell align="center">Preview</TableCell>
                            <TableCell align="center">Inserts Number</TableCell>
                            <TableCell align="center">AI Recognition</TableCell>
                            <TableCell align="center">
                                Show Tags
                                <Checkbox checked={allChecked} onClick={() => handleChangeAllShowMark(!!allChecked)} />
                            </TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {usedBlocks?.map((block) => (
                            <Row
                                key={block.handle1}
                                block={block}
                                expanded={expandedBlocks.includes(block.handle1)}
                                onToggle={() => toggleExpand(block.handle1)}
                            />
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    )
})

export default CurCom