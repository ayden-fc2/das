import React, {useEffect} from 'react';
import {SvgProps} from "@/app/components/draw/SvgGetter";

const SvgRender: React.FC<SvgProps> = (svgProps: SvgProps) => {
    const svgRef = React.useRef<HTMLDivElement>(null);


    // 支持svgString参数，直接渲染svg字符串
    useEffect(() => {
        if (svgProps && svgRef.current) {
            svgRef.current.innerHTML = svgProps.svgString;
        }
    }, [svgProps]);


    return (
        <div ref={svgRef} className="w-full h-full relative flex justify-center items-center">

        </div>
    );
}

export default SvgRender;