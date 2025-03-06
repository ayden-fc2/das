const handleInsert = (xPointIn: number, yPointIn: number, insertScale: number[], insertRotation: number, insertInsPt: number[], debug: boolean = false): number[] => {
    let xPointOut = xPointIn
    let yPointOut = yPointIn
    // 处理缩放
    xPointOut = xPointOut * insertScale[0];
    yPointOut = yPointOut * insertScale[1];
    if (debug) {
        // console.log(xPointOut, '缩放');
    }
    // 处理旋转
    const rotate = insertRotation;  // 旋转角度 (弧度)
    // 保存原始坐标
    const originalX = xPointOut;
    const originalY = yPointOut;
    xPointOut = originalX * Math.cos(rotate) - originalY * Math.sin(rotate);
    yPointOut = originalX * Math.sin(rotate) + originalY * Math.cos(rotate);
    if (debug) {
        // console.log(xPointOut, '旋转');
    }
    // 处理平移
    xPointOut = xPointOut + insertInsPt[0]
    yPointOut = yPointOut + insertInsPt[1]
    if (debug) {
        // console.log(xPointOut, '平移');
    }
    return [xPointOut, yPointOut]
}

export const drawLine = (ctx: CanvasRenderingContext2D, lines: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[]) => {
    // 处理insert
    let handledLines = JSON.parse(JSON.stringify(lines ? lines : []));
    if (insertInsPt && insertRotation !== undefined && insertScale) {
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
    if (insertInsPt && insertRotation !== undefined && insertScale) {
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

export const drawCircle = (ctx: CanvasRenderingContext2D, circles: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[], debug: boolean = false) => {
    // 处理insert
    if (debug) {
        // console.log(circles, '处理前的圆形');
    }
    let handledCircles = JSON.parse(JSON.stringify(circles ? circles : []));
    if (insertInsPt && insertRotation !== undefined && insertScale) {
        if (debug) {
            // console.log('处理圆形');
            // console.log(insertScale, insertRotation, insertInsPt,'缩放、旋转、平移计算圆心和半径')
        }
        for (const circle of handledCircles) {
            // 处理圆心（平移、旋转、缩放）
            [circle.center[0], circle.center[1]] = handleInsert(
                circle.center[0],
                circle.center[1],
                insertScale,
                insertRotation,
                insertInsPt,
                debug
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
    }
    if (debug) {
        console.log(handledCircles, '处理后的圆形');
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

// 绘制曲线Arc
export const drawArc = (ctx: CanvasRenderingContext2D, arcs: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[])=> {
    let handledArcs = JSON.parse(JSON.stringify(arcs ? arcs : []));
    if (insertInsPt && insertRotation !== undefined && insertScale) {
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

// 新增的字符串处理方法
const processMtext = (text: string): string => {
    // 处理控制块：{\...}，提取最后一个分号后的内容
    let processed = text.replace(/\{\\.*?\}/g, (match) => {
        const innerContent = match.slice(1, -1); // 去除最外层花括号
        const lastSemicolonIndex = innerContent.lastIndexOf(';');
        return lastSemicolonIndex !== -1
            ? innerContent.slice(lastSemicolonIndex + 1)
            : innerContent;
    });

    // 处理其他控制符（如\\W0.8; \\pxqc;）
    processed = processed.replace(/\\[^;]*(;|$)/g, '');

    // 处理换行符（保留换行标记供后续split使用）
    processed = processed.replace(/\\P/g, '\n');

    return processed;
};

// 绘制文字MTEXT
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
            mtext.rotatation = insertRotation;
        }
    }


    // 遍历每个 MTEXT 实体绘制
    for (const mtext of handledMtexts) {
        ctx.save();
        // 将画布原点移动到文字的插入点
        ctx.translate(mtext.ins_pt[0], mtext.ins_pt[1]);
        if (mtext.rotatation) {
            ctx.rotate(mtext.rotatation); // 旋转角度，以弧度为单位
        }
        ctx.scale(1, -1);

        // 设置字体（优先从控制块提取）
        const fontMatch = mtext.text.match(/\\f([^|;]+)/);
        const fontFamily = fontMatch?.[1] || 'sans-serif';
        ctx.font = `${mtext.text_height}px ${fontFamily}`;

        // 设置颜色和对齐
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        const colorMatch = mtext.text.match(/\\C(\d+)/);
        ctx.fillStyle = colorMatch
            ? `#${parseInt(colorMatch[1], 10).toString(16).padStart(6, '0')}`
            : `#${mtext.color?.rgb || '000000'}`;

        // 核心文本处理
        const plainText = processMtext(mtext.text);
        const lines = plainText.split('\n');
        const lineHeight = mtext.text_height * (mtext.linespace_factor || 1);

        // 逐行绘制文字
        for (let i = 0; i < lines.length; i++) {
            ctx.fillText(lines[i], 0, i * lineHeight);
        }
        ctx.restore();
    }
}

export const drawText = (ctx: CanvasRenderingContext2D, texts: any, scale: number, insertInsPt?: number[], insertRotation?: number, insertScale?: number[], asMark:boolean = false) => {
    let handledTexts = JSON.parse(JSON.stringify(texts ? texts : []));
    if (insertInsPt && insertRotation !== undefined && insertScale) {
        for (const text of handledTexts) {
            // 处理插入点：平移、旋转、缩放（使用统一的 handleInsert 方法）
            [text.ins_pt[0], text.ins_pt[1]] = handleInsert(
                text.ins_pt[0],
                text.ins_pt[1],
                insertScale,
                insertRotation,
                insertInsPt
            );
            // 处理文字高度（仅缩放）
            text.height = text.height * insertScale[0];
            text.rotatation = insertRotation;
        }
    }

    // 遍历每个 TEXT 实体绘制
    for (const text of handledTexts) {
        ctx.save();
        // 将画布原点移动到文字的插入点
        ctx.translate(text.ins_pt[0], text.ins_pt[1]);
        if (text.rotatation) {
            ctx.rotate(text.rotatation); // 旋转角度，以弧度为单位
        }
        ctx.scale(1, -1);


        // 设置文本对齐方式（可根据需要调整）
        // @ts-ignore
        ctx.textBaseline = "center";
        ctx.textAlign = "left";
        // 设置文字颜色
        ctx.fillStyle = "#" + text.color.rgb;

        // 根据 text_height 和文字格式设置字体（默认使用 sans-serif）
        let fontFamily = "sans-serif";
        ctx.font = `${text.height}px ${fontFamily}`;

        // 本例采用正则，将类似 {\\fSimSun|b0|i0|c134|p2;测试} 替换为 "测试"
        let plainText = processText(text.text_value)

        // // 计算文本宽度
        // let textWidth = ctx.measureText(plainText).width;
        // let textHeight = text.height; // 假设 text.height 代表字体大小

        let offsetX = 0, offsetY = 0;


        ctx.fillText(plainText, offsetX, offsetY);
        ctx.restore();
    }
}

// 新增TEXT处理函数
const processText = (text: string): string => {
    // 处理AutoCAD特殊字符
    let processed = text
        // 直径符号 %%C → Ø
        .replace(/%%[Cc]/g, 'Ø')
        // 度符号 %%D → °
        .replace(/%%D/g, '°')
        // 公差符号 %%P → ±
        .replace(/%%P/g, '±')
        // 百分比 %%O → 上划线（暂不处理）
        // 百分比 %%U → 下划线（暂不处理）
        // 单个%号（如%%%% → %%）
        .replace(/%%/g, '%');

    return processed;
};