import { get } from './api'

export const getJsonObj = async (jsonPath: string)=>{
    get(jsonPath).then(res=>{
        console.log(res)
        return null
    }).catch(err=>{
        console.error(err.message)
        return null
    })
}