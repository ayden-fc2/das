import {get} from "@/app/utils/api";

export function convertToChinaTime(utcDateStr: string): string {
    // 创建一个 UTC 时间对象
    const utcDate = new Date(utcDateStr);

    // 获取中国时间 (UTC + 8)
    const chinaTime = new Date(utcDate.getTime() + 8 * 60 * 60 * 1000); // 转换成 UTC + 8 时区时间

    // 格式化为所需的时间格式，例如：'YYYY-MM-DD HH:mm:ss'
    const year = chinaTime.getFullYear();
    const month = (chinaTime.getMonth() + 1).toString().padStart(2, '0');
    const day = chinaTime.getDate().toString().padStart(2, '0');
    const hours = chinaTime.getHours().toString().padStart(2, '0');
    const minutes = chinaTime.getMinutes().toString().padStart(2, '0');
    const seconds = chinaTime.getSeconds().toString().padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

export async function handleDownload(url: string){
    const result = await get(url) as any;
    const name: string = url.split('/').pop() || 'unknown.dwg'
    // 将二进制数据处理为Blob对象
    const blob = new Blob([result], { type: 'application/octet-stream' });

    // 创建一个下载链接
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = name;

    // 模拟点击下载链接
    link.click();
}