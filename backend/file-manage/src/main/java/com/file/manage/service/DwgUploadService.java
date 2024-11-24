package com.file.manage.service;

import org.springframework.web.multipart.MultipartFile;

public interface DwgUploadService {

    String uploadDwg(MultipartFile file);

    String uploadFile(MultipartFile file);
}
