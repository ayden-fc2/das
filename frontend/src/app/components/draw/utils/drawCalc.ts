// 计算图形包围盒 minX, minY, maxX, maxY

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

export const calcComBox = (typeProps: ComTypeProps) => {
    if (typeProps.CIRCLE.length === 0 && typeProps.LINE.length === 0 && typeProps.LWPOLYLINE.length === 0 && typeProps.ARC.length === 0) {
        return {
            minX: 0,
            minY: 0,
            maxX: 0,
            maxY: 0
        }
    }
    const boxs: Box[] = [];
    for (const lineElement of typeProps.LINE) {
        boxs.push(getLineBox(lineElement));
    }
    for (const circleElement of typeProps.CIRCLE) {
        boxs.push(getCircleBox(circleElement));
    }
    for (const lwpolylineElement of typeProps.LWPOLYLINE) {
        boxs.push(getLwpolylineBox(lwpolylineElement));
    }
    for (const arcElement of typeProps.ARC) {
        boxs.push(getArcBox(arcElement));
    }
    const result = boxs.reduce((acc, box) => ({
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
    return result;
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
