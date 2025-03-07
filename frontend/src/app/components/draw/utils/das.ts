import { get } from '../../../utils/api'
import mockData from '@/app/mock/bbb-handled.json';

const getJsonPath = (jsonPath: string) => {
    // TODO: 配合frp
    if (window.location.hostname !== "localhost") {
        return jsonPath.replace("localhost:", "www.fivecheers.com:");
    }
    return jsonPath;
}

export const getJsonObj = async (jsonPath: string)=>{

    try {
        const res = await get(getJsonPath(jsonPath));
        console.log(res);
        return resolveSeries(res);
    } catch (e:any) {
        console.error(e.message);
        return null;
    }
    // return Promise.resolve(
    //     mockData
    // )
}

const resolveSeries = (res:any):any => {
    let result = {
        TYPES: {
            LINE: [],
            CIRCLE: [],
            POINT: [],
            LWPOLYLINE: [],
            MTEXT: [],
            ARC: [],
            INSERTS: [],
            TEXT: [],
        },
        USED_BLOCKS: res.USED_BLOCKS,
    }
    for (let item of res.MODEL_SPACE) {
        // 筛选INSERTS
        if (item.$ref) {
            // console.log(item.$ref)
            const refPath = item.$ref.split('.')
            const blockIndex = parseInt(refPath[1].match(/\d+/)[0]);  // 获取 USED_BLOCKS[0] 中的 0
            const insertIndex = parseInt(refPath[2].match(/\d+/)[0]); // 获取 inserts[0] 中的 0

            // @ts-ignore
            result.TYPES.INSERTS.push({
                blockIndex: blockIndex,
                insertIndex: insertIndex
            })
        } else if (item.entity === 'INSERT') {
            console.log('INSERT bot ref warning!')
        }
        // 筛选其他entity
        switch (item.entity) {
            case 'LINE':
                // @ts-ignore
                result.TYPES.LINE.push(item);
                break;
            case 'CIRCLE':
                // @ts-ignore
                result.TYPES.CIRCLE.push(item);
                break;
            case 'POINT':
                // @ts-ignore
                result.TYPES.POINT.push(item);
                break;
            case 'LWPOLYLINE':
                // @ts-ignore
                result.TYPES.LWPOLYLINE.push(item);
                break;
            case 'MTEXT':
                // @ts-ignore
                result.TYPES.MTEXT.push(item);
                break;
            case 'ARC':
                // @ts-ignore
                result.TYPES.ARC.push(item);
                break;
            case 'TEXT':
                // @ts-ignore
                result.TYPES.TEXT.push(item);
        }
    }
    // 处理USED_BLOCKS中entity分类
    for (let block of result.USED_BLOCKS) {
        block.showMark = true
        var randomColor = require('randomcolor')
        const color =randomColor(
            {
                luminosity: 'dark',
                hue: 'random',
            }
        ).slice(1) + 'aa'
        block.markColor = color
        block.original_entities.TYPES = {
            LINE: [],
            CIRCLE: [],
            POINT: [],
            LWPOLYLINE: [],
            MTEXT: [],
            ARC: [],
            TEXT: [],
            INSERTS: [],
        }
        for (let item of block.original_entities) {
            switch (item.entity) {
                case 'LINE':
                    block.original_entities.TYPES.LINE.push(item);
                    break;
                case 'CIRCLE':
                    block.original_entities.TYPES.CIRCLE.push(item);
                    break;
                case 'POINT':
                    block.original_entities.TYPES.POINT.push(item);
                    break;
                case 'LWPOLYLINE':
                    block.original_entities.TYPES.LWPOLYLINE.push(item);
                    break;
                case 'MTEXT':
                    block.original_entities.TYPES.MTEXT.push(item);
                    break;
                case 'ARC':
                    block.original_entities.TYPES.ARC.push(item);
                    break;
                case 'TEXT':
                    block.original_entities.TYPES.TEXT.push(item);
                    break;
                case 'INSERT':
                    block.original_entities.TYPES.INSERTS.push(item);
                    break;
            }
        }
    }
    return result
}