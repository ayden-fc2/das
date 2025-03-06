package com.dwg.handler.config;

import com.example.common.enums.UserType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;
import org.springframework.security.oauth2.config.annotation.web.configuration.ResourceServerConfigurerAdapter;
import org.springframework.security.oauth2.config.annotation.web.configurers.ResourceServerSecurityConfigurer;
import org.springframework.security.oauth2.provider.token.TokenStore;

@Configuration
@EnableResourceServer
public class ResourceConfig extends ResourceServerConfigurerAdapter {
    @Override
    public void configure(HttpSecurity http) throws Exception {
        http
                .authorizeRequests()
                .antMatchers("/read").hasAnyAuthority(
                        UserType.Observer.getType()
                        )
                .antMatchers("/cop").hasAnyAuthority(
                        UserType.Controller.getType()
                )
                .antMatchers("/cop/genAnalysis").hasAnyAuthority(
                        UserType.SuperManager.getType()
                )
                .anyRequest().authenticated()
                .and()
                .formLogin().disable()
                .csrf().disable(); // 禁用 CSRF，可能需要根据实际情况启用
    }

    @Override
    public void configure(ResourceServerSecurityConfigurer resources) throws Exception {
        resources.tokenStore(jwtTokenStore);
    }

    @Autowired
    TokenStore jwtTokenStore;
}
