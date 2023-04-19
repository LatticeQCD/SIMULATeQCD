//
// Created by sajid Ali on 12/13/21.
//

#pragma once
class Checksum{
public:
    int32_t checksumb;
    int32_t checksuma;
};

void InitializeChecksum(Checksum *ics)
{
    ics->checksuma=0;
    ics->checksumb=0;
};
//######################################################################################################################

uint32_t checksum_crc32_sitedata(const char *ptr_buffer, size_t bytes){

    uint32_t checksum_crc32=0xFFFFFFFF;//0xFFFFFFFF -> hexadecimal integer, its decimal value is 4294967295
    for(size_t i=0;i<bytes;i++){
        char char_buffer=ptr_buffer[i];
        for(size_t j=0;j<8;j++){
            uint32_t cbORcrc= (char_buffer ^ checksum_crc32) & 1;
            checksum_crc32>>=1;
            if(cbORcrc)
                checksum_crc32= checksum_crc32 ^ 0xEDB88320;
            char_buffer>>=1;
        }
    }
    return ~checksum_crc32;
}
//######################################################################################################################

void checksum_crc32_accumulator(Checksum *checksum_crc32, size_t site_index, char *ptr_buffer, size_t sitedata_bytes){

    size_t site_id29 = site_index;
    size_t site_id31 = site_index;
    //uint32_t cs_crc32_sd = DML_crc32(0, (unsigned char*)ptr_buffer, sitedata_bytes);
    uint32_t cs_crc32_sd = checksum_crc32_sitedata((const char *) ptr_buffer, sitedata_bytes);

    site_id29 %= 29;site_id31 %= 31;

    checksum_crc32->checksuma ^= cs_crc32_sd << site_id29 | cs_crc32_sd >> (32 - site_id29);
    checksum_crc32->checksumb ^= cs_crc32_sd << site_id31 | cs_crc32_sd >> (32 - site_id31);
}
//######################################################################################################################

void checksum_crc32_combine(Checksum  *checksum_crc32,size_t global_vol, uint32_t cs_crc32_sd[]){

    for(size_t i = 0; i < global_vol; i++)
    {
        size_t site_id29 = i;
        size_t site_id31 = i;

        site_id29 %= 29;site_id31 %= 31;

        checksum_crc32->checksuma ^= cs_crc32_sd[i] << site_id29 | cs_crc32_sd[i] >> (32 - site_id29);
        checksum_crc32->checksumb ^= cs_crc32_sd[i] << site_id31 | cs_crc32_sd[i] >> (32 - site_id31);
        //std::cout<<checksum_crc32->checksuma<<"\t"<<checksum_crc32->checksumb<<std::endl;
    }

}
//######################################################################################################################
