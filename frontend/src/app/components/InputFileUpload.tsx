import * as React from 'react';
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import api from "@/app/utils/api";
import {err} from "@/app/utils/alerter";
import {useState} from "react";

const VisuallyHiddenInput = styled('input')({
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: 1,
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    whiteSpace: 'nowrap',
    width: 1,
});

// 定义接口类型
interface InputFileUploadProps {
    maxFiles?: number;  // 最大文件数量，默认为无限制
    acceptTypes?: string;  // 接受的文件类型，默认为所有文件
    apiUrl: string;  // 上传 API 地址
    onSuccess?: (files: string[]) => void;  // 上传成功后的回调函数
}

export default function InputFileUpload({
                                            maxFiles = Infinity,
                                            acceptTypes = '*',
                                            apiUrl,
                                            onSuccess,
                                        }: InputFileUploadProps) {
    const [fileCount, setFileCount] = useState(0);

    // 处理上传
    const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (!files) return;
        if (files.length > maxFiles || fileCount >= maxFiles) {
            err(`You can only upload up to ${maxFiles} files.`);
            return;
        }
        const responses = [];
        for (let i = 0; i < files.length; i++) {
            const formData = new FormData();
            formData.append('file', files[i]);
            responses.push(await api.post(apiUrl, formData));
        }
        Promise.all(responses).then(responses => {
            const allSuccess = responses.every(response => response.data.success === 1);
            if (allSuccess) {
                console.log('All files uploaded successfully');
                const fileUrls = responses.map(response => response.data.data);
                setFileCount(responses.length);
                if (onSuccess) {
                    onSuccess(fileUrls);
                }
            } else {
                console.error('Some files failed to upload');
            }
        }).catch(error => {
            console.error('Error uploading files:', error);
        })
    };

    return (
        <div>
            <Button
                component="label"
                variant="contained"
                startIcon={<CloudUploadIcon />}
            >
                Upload files
                <VisuallyHiddenInput
                    type="file"
                    onChange={handleUpload}
                    multiple
                    accept={acceptTypes}
                />
            </Button>
            <div className="text-gray-500 text-sm mt-2">{`You have uploaded ${fileCount} files.`}</div>
        </div>
    );
}
