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
    TableRow, Typography
} from "@mui/material"
import React from "react"
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import SvgRender from "@/app/components/draw/ComponentRender";
import ComponentRender from "@/app/components/draw/ComponentRender";
import {calcComBox} from "@/app/components/draw/utils/drawCalc";

interface CurComProps {
    usedBlocks: any[],
    changeShowMark: (handle: string) => void
}

const CurCom = ({usedBlocks, changeShowMark}: CurComProps)=> {

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

    function Row(props: any) {
        const { block } = props;
        const [open, setOpen] = React.useState(false);

        return (
            <React.Fragment>
                <TableRow>
                    {/*收缩展开*/}
                    <TableCell>
                        <IconButton
                            aria-label="expand row"
                            size="small"
                            onClick={() => setOpen(!open)}>
                            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                        </IconButton>
                    </TableCell>
                    {/*  Name  */}
                    <TableCell align="center">{block.name}</TableCell>
                    {/*  Preview  */}
                    <TableCell align="center">
                        <Box className={`w-full flex justify-center items-center`}>
                            <Box className={`w-28 h-24 overflow-hidden flex justify-center items-center relative`}>
                                <ComponentRender {...block.original_entities.TYPES} />
                            </Box>
                        </Box>
                    </TableCell>
                    {/*  Inserts Number  */}
                    <TableCell align="center">{block.inserts.length}</TableCell>
                    {/*  Type Inference  */}
                    <TableCell align="center">TODO</TableCell>
                    {/*  Show Tags  */}
                    <TableCell align="center">
                        <Checkbox checked={block.showMark} onChange={() => changeShowMark(block.handle)} />
                    </TableCell>
                </TableRow>
                {/*  展开内容  */}
                <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
                        <Collapse in={open} timeout="auto" unmountOnExit>
                            <Box sx={{ margin: 1 }}>
                                <Typography gutterBottom component="div">
                                    Inserts:
                                </Typography>
                                {/*  Inserts Table  */}
                                <Table size="small" aria-label="purchases">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell align="center">Index</TableCell>
                                            <TableCell align="center">Center Position</TableCell>
                                            <TableCell align="center">Upstream</TableCell>
                                            <TableCell align="center">Downstream</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {block.inserts.map((insert: any, index: number) => (
                                            <TableRow key={index}>
                                                <TableCell align="center">{index}</TableCell>
                                                <TableCell align="center">TODO</TableCell>
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
    }

    return (
        <Box className={`w-full h-full text-black overscroll-y-auto overflow-x-hidden`}>
            <Box className={`h-7 mb-2 flex px-2 items-center justify-center `}>
                <h1 className="text-blue-500 font-bold text-xl">Current DCS Components</h1>
                <span className="ml-2 text-gray-500">( total: {usedBlocks?.length} )</span>
            </Box>
            {/*<Table sx={{ minWidth: 650 }} aria-label="simple table">*/}
            {/*    <TableHead>*/}
            {/*        <TableRow>*/}
            {/*            <TableCell>Name</TableCell>*/}
            {/*            <TableCell>Preview</TableCell>*/}
            {/*            <TableCell>Inserts Number</TableCell>*/}
            {/*            <TableCell>Type Inference</TableCell>*/}
            {/*            <TableCell>Show Tags</TableCell>*/}
            {/*        </TableRow>*/}
            {/*    </TableHead>*/}
            {/*    <TableBody>*/}
            {/*        {usedBlocks?.map((block) => (*/}
            {/*            <TableRow*/}
            {/*                key={block.handle}*/}
            {/*            >*/}
            {/*                <TableCell>*/}
            {/*                    <span className="font-bold">{block.name}</span>*/}
            {/*                </TableCell>*/}
            {/*                <TableCell>*/}
            {/*                    <div className={`w-20 h-20 -my-2 bg-gray-200`}>*/}

            {/*                    </div>*/}
            {/*                </TableCell>*/}
            {/*                <TableCell>{block.inserts.length}</TableCell>*/}
            {/*                <TableCell>TODO</TableCell>*/}
            {/*                <TableCell>TODO</TableCell>*/}
            {/*            </TableRow>*/}
            {/*        ))}*/}
            {/*    </TableBody>*/}
            {/*</Table>*/}

            <TableContainer component={Paper}>
                <Table aria-label="collapsible table">
                    <TableHead>
                        <TableRow>
                            <TableCell align="center" />
                            <TableCell align="center">Name</TableCell>
                            <TableCell align="center">Preview</TableCell>
                            <TableCell align="center">Inserts Number</TableCell>
                            <TableCell align="center">AI Recognition</TableCell>
                            <TableCell align="center">Show Tags</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {usedBlocks?.map((block) => (
                            <Row key={block.handle} block={block} />
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    )
}

export default CurCom