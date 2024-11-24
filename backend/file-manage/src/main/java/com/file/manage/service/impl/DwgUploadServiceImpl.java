package com.file.manage.service.impl;

import com.example.common.exception.MyException;
import com.file.manage.service.DwgUploadService;
import com.file.manage.service.thumbnailator.ImageUtil;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

@Service
public class DwgUploadServiceImpl implements DwgUploadService {

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    @Value("${custom.back-file-path-dwg}" + "/")
    private String dwgReturnUrl;

    @Value("${custom.save-file-path-common}" + "/")
    private String commonUploadFilePath;

    @Value("${custom.back-file-path-common}" + "/")
    private String commonReturnUrl;

    @Override
    public String uploadDwg(MultipartFile file) {
        try {
            String originalFileName = file.getOriginalFilename();
            String fileExtension = originalFileName.substring(originalFileName.lastIndexOf(".") + 1);
            String newFilename = UUID.randomUUID().toString() + "." + fileExtension;
            // 指定绝对路径
            String saveDirectory = dwgUploadFilePath;
            Path savePath = Paths.get(saveDirectory, newFilename); // 使用 Paths.get 创建路径
            Files.createDirectories(savePath.getParent());
            System.out.println(savePath);
            try (InputStream inputStream = file.getInputStream()) {
                Files.copy(inputStream, savePath, StandardCopyOption.REPLACE_EXISTING);
            }
            String baseUrl = dwgReturnUrl;
            String fileUrl = baseUrl + newFilename;
            return fileUrl;
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }

    @Override
    public String uploadFile(MultipartFile file) {
        try {
            String originalFileName = file.getOriginalFilename();
            String fileExtension = originalFileName.substring(originalFileName.lastIndexOf(".") + 1);
            if (fileExtension.equals("png")||fileExtension.equals("jpg")){
                file = ImageUtil.compressImage(400 * 1024, file);
            }
            String newFilename = UUID.randomUUID().toString() + "." + fileExtension;
            // 指定绝对路径
            String saveDirectory = commonUploadFilePath + fileExtension + "/";
            Path savePath = Paths.get(saveDirectory, newFilename); // 使用 Paths.get 创建路径
            Files.createDirectories(savePath.getParent());
            try (InputStream inputStream = file.getInputStream()) {
                Files.copy(inputStream, savePath, StandardCopyOption.REPLACE_EXISTING);
            }
            String baseUrl = commonReturnUrl + fileExtension + "/";
            if (fileExtension.equals("mp4")){
                //使用ffmpeg将file压缩为720p,savePsth下的文件压缩后同名文件放置在savePath/720p/下
                // 定义压缩后的视频存放路径
                String compressedDirectory = saveDirectory + "720p/";
                Path compressedSavePath = Paths.get(compressedDirectory, newFilename); // 使用 Paths.get 创建压缩视频的路径
                Files.createDirectories(compressedSavePath.getParent());
                // 构建ffmpeg命令来压缩视频
                String ffmpegCmd = "ffmpeg -i " + savePath.toString() + " -s hd720 -c:v libx264 -crf 23 -c:a aac -strict -2 " + compressedSavePath.toString();
                // 执行ffmpeg命令
                Process process = Runtime.getRuntime().exec(new String[] { "bash", "-c", ffmpegCmd });
                baseUrl = commonReturnUrl + fileExtension + "/720p/";
                //清空缓存区，保证exe执行
                new Thread(() -> {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                        String line;
                        while ((line = br.readLine()) != null) {
                            System.out.println(line);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }).start();
            }
            String fileUrl = baseUrl + newFilename;
            return fileUrl;
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }
}
