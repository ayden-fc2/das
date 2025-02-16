import { get } from './api'
import mockData from '@/app/mock/bbb-handled.json';


export const getJsonObj = async (jsonPath: string)=>{
    try {
        const res = await get(jsonPath);
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
            POINT: []
        }
    }
    for (let item of res.MODEL_SPACE) {
        switch (item.entity) {
            case 'LINE':
                result.TYPES.LINE.push(item);
                break;
            case 'CIRCLE':
                result.TYPES.CIRCLE.push(item);
                break;
            case 'POINT':
                result.TYPES.POINT.push(item);
                break;
        }
    }
    return result
}