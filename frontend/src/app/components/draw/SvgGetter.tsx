import React, {useEffect} from 'react';
import {calcComBox, ComTypeProps} from "@/app/components/draw/utils/drawCalc";

export interface SvgProps {
    svgString: string;
    svgWidth?: number;
    svgHeight?: number;
}

interface SvgGetterProps {
    comTypeProps: ComTypeProps;
    onSvgStringGenerated?: (svgProps: SvgProps) => void;
}

/**
 * 元器件渲染器，传入 comTypeProps 直接渲染，回调获取svg属性
 * @param comTypeProps
 * @param onSvgStringGenerated
 * @constructor
 */
const SvgGetter: React.FC<SvgGetterProps> = ({ comTypeProps, onSvgStringGenerated }) => {
    const svgRef = React.useRef<SVGSVGElement>(null);

    const getSvgWidth = (comTypeProps: ComTypeProps) => {
        const { minX, maxX } = calcComBox(comTypeProps)
        return isFinite(maxX - minX + 2) ? maxX - minX + 2 : 2
    }

    const getSvgHeight = (comTypeProps: ComTypeProps) => {
        const { minY, maxY } = calcComBox(comTypeProps)
        return isFinite(maxY - minY + 2) ? maxY - minY + 2 : 2
    }

    const getSvgElements = (comTypeProps: ComTypeProps) => {

        const { minX, maxX, minY, maxY } = calcComBox(comTypeProps)
        const svgWidth = getSvgWidth(comTypeProps)
        const svgHeight = getSvgHeight(comTypeProps)
        const getLineWidth = (svgWidth: number, svgHeight: number)=> {
            return Math.max(svgWidth, svgHeight) / 60;
        }

        // 辅助坐标转换函数
        const correctPoint = (x: number, y: number, yReverse = true) => {
            if (yReverse) {
                return [x - minX + 1, svgHeight - (y - minY) - 1]
            }
            return [x - minX, y - minY];
        }


        let svgElements = [];
        for (const circle of comTypeProps.CIRCLE) {
            svgElements.push(
                <circle
                    key={circle.handle}
                    cx={correctPoint(circle.center[0], circle.center[1])[0]}
                    cy={correctPoint(circle.center[0], circle.center[1])[1]}
                    r={circle.radius}
                    fill="none"
                    stroke="black"
                    strokeWidth={getLineWidth(svgWidth, svgHeight)}
                />
            );
        }
        for (const line of comTypeProps.LINE) {
            svgElements.push(
                <line
                    key={line.handle}
                    x1={correctPoint(line.start[0], line.start[1])[0]}
                    y1={correctPoint(line.start[0], line.start[1])[1]}
                    x2={correctPoint(line.end[0], line.end[1])[0]}
                    y2={correctPoint(line.end[0], line.end[1])[1]}
                    fill="none"
                    stroke="black"
                    strokeWidth={getLineWidth(svgWidth, svgHeight)}
                />
            )
        }

        for (const lwpolyline of comTypeProps.LWPOLYLINE) {
            if (lwpolyline.flag !== 0 && lwpolyline.points[0] !== lwpolyline.points[lwpolyline.points.length - 1]) {
                lwpolyline.points.push(lwpolyline.points[0]);
            }
            let drawPoints =
                svgElements.push(
                    <polyline
                        key={lwpolyline.handle}
                        points={lwpolyline.points.map((point: number[]) => correctPoint(point[0], point[1]).join(',')).join(' ')}
                        fill="none"
                        stroke="black"
                        strokeWidth={getLineWidth(svgWidth, svgHeight)}
                    />
                )
        }

        for (const arc of comTypeProps.ARC) {
            const startAngle = arc.start_angle;
            const endAngle = arc.end_angle;
            const radius = arc.radius;
            const [cx, cy] = arc.center; // 圆心坐标

            // 计算起点坐标（考虑SVG的Y轴向下）
            const startX = cx + radius * Math.cos(startAngle);
            const startY = cy + radius * Math.sin(startAngle); // 使用减法翻转Y轴

            // 计算终点坐标
            const endX = cx + radius * Math.cos(endAngle);
            const endY = cy + radius * Math.sin(endAngle);

            // 计算角度差并调整到[0, 2π)范围
            let deltaTheta = endAngle - startAngle;
            if (deltaTheta < 0) {
                deltaTheta += 2 * Math.PI;
            }

            // 判断是否为大弧和绘制方向
            const largeArcFlag = deltaTheta > Math.PI ? 1 : 0;
            const sweepFlag = 0 // 方向由角度差正负决定

            // 构建路径字符串
            const pathData = `M ${correctPoint(startX, startY)[0]} ${correctPoint(startX, startY)[1]} A ${radius} ${radius} 0 ${largeArcFlag} ${sweepFlag} ${correctPoint(endX, endY)[0]} ${correctPoint(endX, endY)[1]}`;

            svgElements.push(
                <path
                    key={arc.handle}
                    d={pathData}
                    fill="none"
                    stroke="black"
                    strokeWidth={getLineWidth(svgWidth, svgHeight)}
                />
            );
        }
        return svgElements;
    }

    // 渲染后将生成的 svg 字符串传递给外部（例如存储到后端）
    useEffect(() => {
        if (svgRef.current && onSvgStringGenerated) {
            const generatedSvgString = new XMLSerializer().serializeToString(svgRef.current);
            onSvgStringGenerated({
                svgString: generatedSvgString,
                svgWidth: getSvgWidth(comTypeProps),
                svgHeight: getSvgHeight(comTypeProps)
            });
        }
    }, [comTypeProps, onSvgStringGenerated]);

    return (
        <div className="w-full h-full relative flex justify-center items-center">
            <svg
                ref={svgRef}
                xmlns="http://www.w3.org/2000/svg"
                width="100%"
                height="100%"
                viewBox={`0 0 ${getSvgWidth(comTypeProps)} ${getSvgHeight(comTypeProps)}`}
            >
                { getSvgElements(comTypeProps) }
            </svg>
            {/*<button onClick={handleSave}>保存</button>*/}
        </div>
    );
}

export default SvgGetter;