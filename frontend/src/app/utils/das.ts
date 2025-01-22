import { get } from './api'
import mockData from '@/app/mock/ccc-handled.json';


export const getJsonObj = async (jsonPath: string)=>{
    // get(jsonPath).then(res=>{
    //     console.log(res)
    //     return null
    // }).catch(err=>{
    //     console.error(err.message)
    //     return null
    // })
    return Promise.resolve(
        mockData
    )
}