"use client"

import {
    Box, Button, Checkbox,
    Collapse,
    IconButton,
    List, ListItem, ListItemText,
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
import {block} from "sharp";

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
        const [filteredHandles, setFilteredHandles] = React.useState<string[] | null>(null);

        const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            const input = e.target.value;
            setFilterInput(input);
            // 解析输入为数字数组，过滤无效值
            const handles = input
                .split(' ')
            setFilteredHandles(handles.length > 0 ? handles : null);
        };

        // 生成过滤后的插入项列表
        const { filteredInserts, remainingInserts } = block.inserts
            .map((insert: any, originalIndex: number) => ({ insert, originalIndex }))
            .reduce(
                (acc:any, { insert, originalIndex }: any) => {
                    if (!filteredHandles) {
                        acc.remainingInserts.push({ insert, originalIndex });
                    } else {
                        const h = insert.handle0 + '-' + insert.handle1;
                        if (filteredHandles.includes(h)) {
                            acc.filteredInserts.push({ insert, originalIndex });
                        } else {
                            acc.remainingInserts.push({ insert, originalIndex });
                        }
                    }
                    return acc;
                },
                { filteredInserts: [], remainingInserts: [] } as {
                    filteredInserts: { insert: any; originalIndex: number }[];
                    remainingInserts: { insert: any; originalIndex: number }[];
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
                                            <TableCell align="center">Operation</TableCell>
                                            <TableCell align="center">Upstream</TableCell>
                                            <TableCell align="center">Downstream</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {filteredInserts.map(({ insert, originalIndex }: {insert: any, originalIndex: any}) => (
                                            getCenterPt(insert.centerX, insert.centerY) &&
                                            <TableRow key={originalIndex} className={`bg-red-100`}>
                                                <TableCell align="center">{insert.handle0}-{insert.handle1}</TableCell>
                                                <TableCell align="center">{getCenterPt(insert.centerX, insert.centerY)}</TableCell>
                                                <TableCell align="center">
                                                    <Button
                                                        className="w-full"
                                                        onClick={() => focusInnerClick([insert.centerX, insert.centerY], Math.max(insert.boxWidth, insert.boxHeight))}
                                                        size="small">
                                                        focus
                                                    </Button>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <List dense={true}>
                                                        {generateStream(insert.upstream)}
                                                    </List>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <List dense={true}>
                                                        {generateStream(insert.downstream)}
                                                    </List>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                        {remainingInserts.map(({ insert, originalIndex }: {insert: any, originalIndex: any}) => (
                                            getCenterPt(insert.centerX, insert.centerY) &&
                                            <TableRow key={originalIndex}>
                                                <TableCell align="center">{insert.handle0}-{insert.handle1}</TableCell>
                                                <TableCell align="center">{getCenterPt(insert.centerX, insert.centerY)}</TableCell>
                                                <TableCell align="center">
                                                    <Button
                                                        className="w-full"
                                                        onClick={() => focusInnerClick([insert.centerX, insert.centerY], Math.max(insert.boxWidth, insert.boxHeight))}
                                                        size="small">
                                                        focus
                                                    </Button>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <List dense={true}>
                                                        {generateStream(insert.upstream)}
                                                    </List>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <List dense={true}>
                                                        {generateStream(insert.downstream)}
                                                    </List>
                                                </TableCell>
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

    // 生成List
    function generateStream(upstream: any) {
        const items = upstream.split(',');
        return items.map((item: any, index: number) => {
            if(item) {
                const blockHandle0 = item.split('-')[0]
                const blockHandle1 = item.split('-')[1]
                let blockName = ''
                let insert = null
                for (const b of usedBlocks) {
                    for (const i of b.inserts) {
                        if (i.handle0 === parseInt(blockHandle0) && i.handle1 === parseInt(blockHandle1)) {
                            blockName = b.blockName
                            insert = i
                            break
                        }
                    }
                }
                return (
                    <ListItem key={index}>
                        <Button className="w-full" size="small" onClick={() => focusInnerClick([insert.centerX, insert.centerY], Math.max(insert.boxWidth, insert.boxHeight))}>{blockName} {item}</Button>
                    </ListItem>
                )
            }
        })
    }

    return (
        <Box className={`w-full h-full text-black overscroll-y-auto overflow-x-hidden`}>
            <Box className={`h-7 mb-2 flex px-2 items-center justify-center `}>
                <h1 className="text-blue-500 font-bold text-xl">Current DCS Components</h1>
                <span className="ml-2 text-gray-500">( total: {usedBlocks?.length} )</span>
            </Box>

            <Box>
                <Table aria-label="collapsible table">
                    <TableHead>
                        <TableRow>
                            <TableCell align="center" />
                            <TableCell align="center">Name</TableCell>
                            <TableCell align="center">Preview</TableCell>
                            <TableCell align="center">Inserts Number</TableCell>
                            <TableCell align="center">Type Inference</TableCell>
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
            </Box>
        </Box>
    )
})

export default CurCom