/**
 * 计算图形包围盒 minX, minY, maxX, maxY
 */
import {handleInsert} from "@/app/components/draw/utils/draw";

export interface ComTypeProps {
    LINE: any[];
    CIRCLE: any[];
    LWPOLYLINE: any[];
    ARC: any[];
}
interface Box {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
}

// TODO: 包围盒围绕中心旋转缩放，更改返回类型，后端同步更新处理
export const calcComBox = (typeProps: ComTypeProps, insertRotation?: number, insertScale?: number[], debug = false) => {
    const CIRCLE = JSON.parse(JSON.stringify(typeProps?.CIRCLE || []));
    const LINE = JSON.parse(JSON.stringify(typeProps?.LINE || []));
    const LWPOLYLINE = JSON.parse(JSON.stringify(typeProps?.LWPOLYLINE || []));
    const ARC = JSON.parse(JSON.stringify(typeProps?.ARC || []));
    // 兜底
    if (CIRCLE.length === 0 && LINE.length === 0 && LWPOLYLINE.length === 0 && ARC.length === 0) {
        return {
            minX: 0,
            minY: 0,
            maxX: 0,
            maxY: 0
        }
    }

    // 计算包围盒
    let boxs: Box[] = [];
    for (const lineElement of LINE) {
        boxs.push(getLineBox(lineElement));
    }
    for (const circleElement of CIRCLE) {
        boxs.push(getCircleBox(circleElement));
    }
    for (const lwpolylineElement of LWPOLYLINE) {
        boxs.push(getLwpolylineBox(lwpolylineElement));
    }
    for (const arcElement of ARC) {
        boxs.push(getArcBox(arcElement));
    }
    let result = boxs.reduce((acc, box) => ({
        minX: Math.min(acc.minX, box.minX),
        maxX: Math.max(acc.maxX, box.maxX),
        minY: Math.min(acc.minY, box.minY),
        maxY: Math.max(acc.maxY, box.maxY)
    }), {
        minX: Infinity,
        maxX: -Infinity,
        minY: Infinity,
        maxY: -Infinity
    });
    // 算两次
    if (insertRotation && insertScale) {
        const centerX = (result.minX + result.maxX) / 2
        const centerY = (result.minY + result.maxY) / 2;
        // if (debug) {
        //     console.log(insertRotation, insertScale, 'aaa')
        // }
        for (const line of LINE ?? []) {
            [line.start[0], line.start[1]] = handleInsert(line.start[0], line.start[1], insertScale, insertRotation, [centerX, centerY]);
            [line.end[0], line.end[1]] = handleInsertFromCenter(line.end[0], line.end[1], insertScale, insertRotation, [centerX, centerY]);
        }
        for (const line of LWPOLYLINE ?? []) {
            for (let i = 0; i < line.points.length; i++) {
                const point = line.points[i];
                const [x, y] = handleInsertFromCenter(point[0], point[1], insertScale, insertRotation, [centerX, centerY], debug);
                line.points[i] = [x, y];
            }
        }
        for (const circle of CIRCLE ?? []) {
            // 处理圆心（平移、旋转、缩放）
            [circle.center[0], circle.center[1]] = handleInsertFromCenter(
                circle.center[0],
                circle.center[1],
                insertScale,
                insertRotation,
                [centerX, centerY]
            );
            // 处理半径：仅进行缩放（假设均匀缩放）
            if (insertScale[0] < 0) {
                circle.radius = -circle.radius;
            }
            if (isNaN(circle.radius)) {
                console.log('NaN warning!!!')
            }
            circle.radius = circle.radius * insertScale[0];
        }
        for (const arc of ARC ?? []) {
            // 1. 处理圆心：平移、旋转、缩放
            [arc.center[0], arc.center[1]] = handleInsertFromCenter(
                arc.center[0],
                arc.center[1],
                insertScale,
                insertRotation,
                [centerX, centerY]
            );
            // 2. 处理半径：仅进行缩放（假设均匀缩放）
            arc.radius = arc.radius * insertScale[0];
            // 3. 处理角度：旋转会影响弧线的起始和终止角度，需加上插入的旋转角度
            arc.start_angle = arc.start_angle + insertRotation;
            arc.end_angle = arc.end_angle + insertRotation;
        }
        // 重新计算包围盒
        boxs = [];
        for (const lineElement of LINE) {
            boxs.push(getLineBox(lineElement));
        }
        for (const circleElement of CIRCLE) {
            boxs.push(getCircleBox(circleElement));
        }
        for (const lwpolylineElement of LWPOLYLINE) {
            boxs.push(getLwpolylineBox(lwpolylineElement));
        }
        for (const arcElement of ARC) {
            boxs.push(getArcBox(arcElement));
        }
        result = boxs.reduce((acc, box) => ({
            minX: Math.min(acc.minX, box.minX),
            maxX: Math.max(acc.maxX, box.maxX),
            minY: Math.min(acc.minY, box.minY),
            maxY: Math.max(acc.maxY, box.maxY)
        }), {
            minX: Infinity,
            maxX: -Infinity,
            minY: Infinity,
            maxY: -Infinity
        });
    }
    // if (debug) {
    //     console.log(result);
    // }
    return result;
}

const handleInsertFromCenter = (x: number, y: number, scale: number[], rotation: number, center: number[], debug=false): [number, number] => {
    // 1. 平移至原点（以center为中心）
    const translatedX = x - center[0];
    const translatedY = y - center[1];

    // 2. 缩放
    const scaledX = translatedX * scale[0];
    const scaledY = translatedY * scale[1];

    // 3. 旋转
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    const rotatedX = scaledX * cos - scaledY * sin;
    const rotatedY = scaledX * sin + scaledY * cos;

    // 4. 平移回原坐标系
    const resultX = rotatedX + center[0];
    const resultY = rotatedY + center[1];

    return [resultX, resultY];
}

const getLineBox = (line: any) => {
    return {
        minX: Math.min(line.start[0], line.end[0],),
        maxX: Math.max(line.start[0], line.end[0]),
        minY: Math.min(line.start[1], line.end[1]),
        maxY: Math.max(line.start[1], line.end[1])
    }
}

const getCircleBox = (circle: any) => {
    return {
        minX: circle.center[0] - circle.radius,
        maxX: circle.center[0] + circle.radius,
        minY: circle.center[1] - circle.radius,
        maxY: circle.center[1] + circle.radius
    }
}

const getLwpolylineBox = (lwpolyline: any)=> {
    let result = {
        minX: lwpolyline.points[0][0],
        maxX: lwpolyline.points[0][0],
        minY: lwpolyline.points[0][1],
        maxY: lwpolyline.points[0][1]
    }
    for (let i = 1; i < lwpolyline.points.length; i += 1) {
        const point = lwpolyline.points[i];
        result.minX = Math.min(result.minX, point[0]);
        result.maxX = Math.max(result.maxX, point[0]);
        result.minY = Math.min(result.minY, point[1]);
        result.maxY = Math.max(result.maxY, point[1]);
    }

    return result;
}

const isAngleInSweep = (theta: number, start: number, end: number): boolean => {
    const normalizedTheta = (theta + Math.PI * 2) % (Math.PI * 2);
    if (start <= end) {
        return normalizedTheta >= start && normalizedTheta <= end;
    }
    return normalizedTheta >= start || normalizedTheta <= end;
};

const getArcBox = (arc: any): Box => {
    const center = arc.center;
    const radius = arc.radius;
    const cx = center[0];
    const cy = center[1];

    // 标准化角度到[0, 2π)范围
    let start = (arc.start_angle % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
    let end = (arc.end_angle % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);

    const candidatePoints: { x: number; y: number }[] = [];

    // 添加起点和终点
    candidatePoints.push({
        x: cx + radius * Math.cos(start),
        y: cy + radius * Math.sin(start)
    });
    candidatePoints.push({
        x: cx + radius * Math.cos(end),
        y: cy + radius * Math.sin(end)
    });

    // 检查四个关键角度（轴对齐方向）
    const keyAngles = [0, Math.PI/2, Math.PI, Math.PI*3/2];
    keyAngles.forEach(theta => {
        if (isAngleInSweep(theta, start, end)) {
            candidatePoints.push({
                x: cx + radius * Math.cos(theta),
                y: cy + radius * Math.sin(theta)
            });
        }
    });

    // 计算包围盒
    return candidatePoints.reduce((acc, point) => ({
        minX: Math.min(acc.minX, point.x),
        maxX: Math.max(acc.maxX, point.x),
        minY: Math.min(acc.minY, point.y),
        maxY: Math.max(acc.maxY, point.y)
    }), {
        minX: candidatePoints[0].x,
        maxX: candidatePoints[0].x,
        minY: candidatePoints[0].y,
        maxY: candidatePoints[0].y
    });
};

// 过滤无效的块
export const checkValidBlock = (block: any) => {
    const boxResult = calcComBox(block.original_entities.TYPES);
    if (boxResult.minX === 0 && boxResult.maxX === 0 && boxResult.minY === 0 && boxResult.maxY === 0) {
        return false
    }
    return true
}