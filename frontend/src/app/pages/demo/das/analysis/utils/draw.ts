const handleInsert = (xPointIn: number, yPointIn: number, insertScale: number[], insertRotation: number, insertInsPt: number[]): number[] => {
    let xPointOut = xPointIn
    let yPointOut = yPointIn
    // 处理缩放
    xPointOut = xPointOut * insertScale[0];
    yPointOut = yPointOut * insertScale[1];
    // 处理旋转
    const rotate = insertRotation;  // 旋转角度 (弧度)
    // 保存原始坐标
    const originalX = xPointOut;
    const originalY = yPointOut;
    xPointOut = originalX * Math.cos(rotate) - originalY * Math.sin(rotate);
    yPointOut = originalX * Math.sin(rotate) + originalY * Math.cos(rotate);
    // 处理平移
    xPointOut = xPointOut + insertInsPt[0]
    yPointOut = yPointOut + insertInsPt[1]
    return [xPointOut, yPointOut]
}

export const drawLine = (ctx: CanvasRenderingContext2D, lines: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[]) => {
    // 处理insert
    let handledLines = JSON.parse(JSON.stringify(lines ? lines : []));
    if (insertInsPt && insertRotation && insertScale) {
        for (const line of handledLines ?? []) {
            [line.start[0], line.start[1]] = handleInsert(line.start[0], line.start[1], insertScale, insertRotation, insertInsPt);
            [line.end[0], line.end[1]] = handleInsert(line.end[0], line.end[1], insertScale, insertRotation, insertInsPt);
        }
    }

    for (const line of handledLines ?? []) {
        ctx.beginPath();
        let lineStartX = line.start[0];
        let lineStartY = line.start[1];
        let lineEndX = line.end[0];
        let lineEndY = line.end[1];

        ctx.moveTo(lineStartX, lineStartY);
        ctx.lineTo(lineEndX, lineEndY);
        ctx.strokeStyle = "#" + line.color.rgb;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        ctx.closePath();
    }
}

export const drawLwpolyLine = (ctx: CanvasRenderingContext2D, lwpolylines: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[]) => {
    // 处理insert
    let handledLwpolylines = JSON.parse(JSON.stringify(lwpolylines ? lwpolylines : []));
    if (insertInsPt && insertRotation && insertScale) {
        for (const line of handledLwpolylines ?? []) {
            for (let i = 0; i < line.points.length; i++) {
                const point = line.points[i];
                const [x, y] = handleInsert(point[0], point[1], insertScale, insertRotation, insertInsPt);
                line.points[i] = [x, y];
            }
        }
    }

    // 绘制折线
    for (const line of handledLwpolylines ?? []) {
        ctx.beginPath();
        ctx.strokeStyle = "#" + line.color.rgb;
        ctx.lineWidth = (1 / scale);

        // 遍历 points 绘制折线
        for (let i = 0; i < line.points.length; i++) {
            const point = line.points[i];

            if (i === 0) {
                ctx.moveTo(point[0], point[1]);
            } else {
                ctx.lineTo(point[0], point[1]);
            }
        }
        // 闭合路径
        if (line.flag !== 0) {
            const firstPoint = line.points[0];
            ctx.lineTo(firstPoint[0], firstPoint[1]);
        }
        ctx.stroke();
        ctx.closePath();
    }
}

export const drawCircle = (ctx: CanvasRenderingContext2D, circles: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[]) => {
    // 处理insert
    let handledCircles = JSON.parse(JSON.stringify(circles ? circles : []));
    if (insertInsPt && insertRotation && insertScale) {
        for (const circle of handledCircles) {
            // 处理圆心（平移、旋转、缩放）
            [circle.center[0], circle.center[1]] = handleInsert(
                circle.center[0],
                circle.center[1],
                insertScale,
                insertRotation,
                insertInsPt
            );
            // 处理半径：仅进行缩放（假设均匀缩放）
            circle.radius = circle.radius * insertScale[0];
        }
    }

    for (const circle of handledCircles ?? []) {
        ctx.beginPath();
        ctx.arc(
            circle.center[0],
            circle.center[1],
            circle.radius,
            0,
            Math.PI * 2
        );
        ctx.strokeStyle = "#" + circle.color.rgb;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        ctx.closePath();
    }
}

// TODO: 绘制曲线Arc
export const drawArc = (ctx: CanvasRenderingContext2D, arcs: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[])=> {
    let handledArcs = JSON.parse(JSON.stringify(arcs ? arcs : []));
    if (insertInsPt && insertRotation && insertScale) {
        for (const arc of handledArcs) {
            // 1. 处理圆心：平移、旋转、缩放
            [arc.center[0], arc.center[1]] = handleInsert(
                arc.center[0],
                arc.center[1],
                insertScale,
                insertRotation,
                insertInsPt
            );
            // 2. 处理半径：仅进行缩放（假设均匀缩放）
            arc.radius = arc.radius * insertScale[0];
            // 3. 处理角度：旋转会影响弧线的起始和终止角度，需加上插入的旋转角度
            arc.start_angle = arc.start_angle + insertRotation;
            arc.end_angle = arc.end_angle + insertRotation;
        }
    }

    // 绘制 ARC（圆弧）
    for (const arc of handledArcs) {
        ctx.beginPath();
        // 使用 canvas arc 方法绘制圆弧：
        // 参数依次为：圆心 x、圆心 y、半径、起始角度、终止角度、绘制方向（true 表示逆时针）
        ctx.arc(
            arc.center[0],
            arc.center[1],
            arc.radius,
            arc.start_angle,
            arc.end_angle,
            false // DXF 中 ARC 默认是逆时针绘制
        );
        ctx.strokeStyle = "#" + arc.color.rgb;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        ctx.closePath();
    }
}

// TODO: 绘制文字MTEXT
export const drawMtext = (ctx: CanvasRenderingContext2D, mtexts: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[]) => {
    let handledMtexts = JSON.parse(JSON.stringify(mtexts ? mtexts : []));
    if (insertInsPt && insertRotation !== undefined && insertScale) {
        for (const mtext of handledMtexts) {
            // 处理插入点：平移、旋转、缩放（使用统一的 handleInsert 方法）
            [mtext.ins_pt[0], mtext.ins_pt[1]] = handleInsert(
                mtext.ins_pt[0],
                mtext.ins_pt[1],
                insertScale,
                insertRotation,
                insertInsPt
            );
            // 处理文字高度（仅缩放）
            mtext.text_height = mtext.text_height * insertScale[0];
        }
    }

    // 遍历每个 MTEXT 实体绘制
    for (const mtext of handledMtexts) {
        ctx.save();
        // 将画布原点移动到文字的插入点
        ctx.translate(mtext.ins_pt[0], mtext.ins_pt[1]);
        ctx.scale(1, -1);

        ctx.strokeStyle = "red";
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, mtext.extents_width, mtext.extents_height);

        // 设置文本对齐方式（可根据需要调整）
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        // 设置文字颜色
        ctx.fillStyle = "#" + mtext.color.rgb;

        // 根据 text_height 和文字格式设置字体（默认使用 sans-serif）
        let fontFamily = "sans-serif";
        // 尝试从 MTEXT.text 中提取字体，如 "{\fSimSun|b0|i0|c134|p2;测试}abc"
        const fontMatch = mtext.text.match(/\\f([^|;]+)/);
        if (fontMatch && fontMatch[1]) {
            fontFamily = fontMatch[1];
        }
        ctx.font = `${mtext.text_height}px ${fontFamily}`;

        // 简单处理 MTEXT 内部格式，提取纯文本
        // 本例采用正则，将类似 {\\fSimSun|b0|i0|c134|p2;测试} 替换为 "测试"
        let plainText = mtext.text.replace(/\{\\[^\{]+\}/g, (match) => {
            const semicolonIndex = match.indexOf(";");
            if (semicolonIndex !== -1) {
                // 去除结尾的 "}"
                return match.substring(semicolonIndex + 1, match.length - 1);
            }
            return "";
        });
        // 去掉残留的花括号
        plainText = plainText.replace(/[{}]/g, "");

        // 处理换行符 "\P"，将文本拆分成多行
        const lines = plainText.split("\\P");
        const lineSpacingFactor = mtext.linespace_factor || 1; // 获取行间距系数
        const lineHeight = mtext.text_height * lineSpacingFactor; // 计算实际行高

        // 逐行绘制文字
        for (let i = 0; i < lines.length; i++) {
            ctx.fillText(lines[i], 0, i * lineHeight);
        }
        ctx.restore();
    }
}