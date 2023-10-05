def angka_ke_kata(angka):
    # Daftar kata dalam bahasa Indonesia untuk angka 0 hingga 9
    kata = ["ᨊᨚᨒᨚ", "ᨔᨙᨉᨗ", "ᨉᨘᨕ", "ᨈᨛᨒᨘ", "ᨕᨛᨄ", "ᨒᨗᨆ", "ᨕᨛᨊᨛ", "ᨄᨗᨈᨘ", "ᨕᨑᨘᨕ", "ᨕᨔᨙᨑ"]

    # Daftar kata khusus untuk angka 10 hingga 19
    kata_khusus = ["ᨔᨛᨄᨘᨒᨚ", "ᨔᨛᨄᨘᨒᨚ ᨔᨙᨉᨗ", "ᨔᨛᨄᨘᨒᨚ ᨉᨘᨕ", "ᨔᨛᨄᨘᨒᨚ ᨈᨛᨒᨘ", "ᨔᨛᨄᨘᨒᨚ ᨕᨛᨄ", "ᨔᨛᨄᨘᨒᨚ ᨒᨗᨆ", "ᨔᨛᨄᨘᨒᨚ ᨕᨛᨊᨛ", "ᨔᨛᨄᨘᨒᨚ ᨄᨗᨈᨘ", "ᨔᨛᨄᨘᨒᨚ ᨕᨑᨘᨕ", "ᨔᨛᨄᨘᨒᨚ ᨕᨔᨙᨑ"]

    # Daftar kata untuk puluhan
    #Terjadi perubahan penyebutan Empat menjadi PATA ketika dia diawal bilangan, bukan akhiran untuk puluhan
    puluhan = ["", "ᨔᨛᨄᨘᨒᨚ", "ᨉᨘᨕᨄᨘᨒᨚ", "ᨈᨛᨒᨘᨄᨘᨒᨚ", "ᨄᨈᨄᨘᨒᨚ", "ᨒᨗᨆᨄᨘᨒᨚ", "ᨕᨛᨊᨛᨄᨘᨒᨚᨊ", "ᨄᨗᨈᨘᨄᨘᨒᨚ", "ᨕᨑᨘᨕᨄᨘᨒᨚᨊ", "ᨕᨔᨙᨑᨄᨘᨒᨚᨊ"]

    # Daftar kata untuk ratusan
    # angka ratus dalam angka 400 berubah menjadi DATU
    #Terjadi penyisipan NA di bilangan 6-8-9, di posisi bilangan Puluhan dan Ratusan, tidak pada Ribuan ke atas
    ratusan = ["", "ᨔᨗᨑᨈᨘ", "ᨉᨘᨕᨑᨈᨘ", "ᨈᨛᨒᨘᨑᨈᨘ", "ᨕᨛᨄᨉᨈᨘ", "ᨒᨗᨆᨑᨈᨘ", "ᨕᨛᨊᨛᨑᨈᨘᨊ", "ᨄᨗᨈᨘᨑᨈᨘ", "ᨕᨑᨘᨕᨑᨈᨘᨊ", "ᨕᨔᨙᨑᨑᨈᨘᨊ"]

    ribuan = ["", "ᨔᨗᨔᨛᨅᨘ", "ᨉᨘᨕᨔᨛᨅᨘ", "ᨈᨛᨒᨘᨔᨛᨅᨘ", "ᨄᨈᨔᨛᨅᨘ", "ᨒᨗᨆᨔᨛᨅᨘ", "ᨕᨛᨊᨛᨔᨛᨅᨘ", "ᨄᨗᨈᨘᨔᨛᨅᨘ", "ᨕᨑᨘᨕᨔᨛᨅᨘ", "ᨕᨔᨙᨑᨔᨛᨅᨘ"]


    # Konversi angka menjadi string
    angka_str = str(angka)

    if angka == 0:
        return kata[0]  # Angka nol

    if angka < 10:
        return kata[angka]  # Angka 1 hingga 9

    if 10 <= angka < 20:
        return kata_khusus[angka - 10]  # Angka 10 hingga 19

    if 20 <= angka < 100:
        puluhan_idx = angka // 10
        satuan_idx = angka % 10
        return puluhan[puluhan_idx] + " " + kata[satuan_idx] if satuan_idx != 0 else puluhan[puluhan_idx]  # Puluhan

    if 100 <= angka < 1000:
        ratusan_idx = angka // 100
        sisanya = angka % 100
        if sisanya == 0:
            return ratusan[ratusan_idx]  # Ratusan tanpa puluhan
        else:
            return ratusan[ratusan_idx] + " " + angka_ke_kata(sisanya)  # Ratusan dengan puluhan

    if 1000 <= angka < 1000000: # ribuan
        ribuan_idx = angka // 1000
        sisanya = angka % 1000
        if sisanya == 0 and ribuan_idx < 10:
            return ribuan[ribuan_idx]  # Ribuan tanpa ratuan
        elif sisanya == 0 and 10 <= ribuan_idx < 1000:
            return angka_ke_kata(ribuan_idx) + " " + "ᨔᨛᨅᨘ"
        else:
            return angka_ke_kata(ribuan_idx) + " " + "ᨔᨛᨅᨘ" + " " + angka_ke_kata(sisanya)  # Ribuan dengan ratusan dan puluhan

    if 1000000 <= angka < 1000000000000000: #juta, miliar, triliun
        if 1000000 <= angka < 1000000000: # jutaan
          jutaan_idx = angka // 1000000
          sisanya = angka % 1000000
          if 1 <= jutaan_idx < 1000:
              if sisanya == 0 :
                  return angka_ke_kata(jutaan_idx) + " " + "ᨍᨘᨈ"
              else:
                  return angka_ke_kata(jutaan_idx) + " " + "ᨍᨘᨈ" + " "+ angka_ke_kata(sisanya)

        elif 1000000000 <= angka < 1000000000000: # miliaran
          miliaran_idx = angka // 1000000000
          sisanya = angka % 1000000000
          if 1 <= miliaran_idx < 1000:
              if sisanya == 0 :
                  return angka_ke_kata(miliaran_idx) + " " + "ᨆᨗᨒᨗᨕᨑᨛ"
              else:
                  return angka_ke_kata(miliaran_idx) + " " + "ᨆᨗᨒᨗᨕᨑᨛ" + " "+ angka_ke_kata(sisanya)
        elif 1000000000000 <= angka < 1000000000000000:
          triliun_idx = angka // 1000000000000 # triliun
          sisanya = angka % 1000000000000
          if 1 <= triliun_idx < 1000:
              if sisanya == 0 :
                  return angka_ke_kata(triliun_idx) + " " + "ᨈᨑᨒᨗᨕᨘ"
              else:
                  return angka_ke_kata(triliun_idx) + " " + "ᨈᨑᨒᨗᨕᨘ" + " "+ angka_ke_kata(sisanya)

    return "ᨔᨛᨊ ᨆᨒᨚᨄᨚ ᨊᨚᨆᨚᨑᨚᨊ"  # Angka lebih dari 1000 (untuk contoh ini)


def number_expand(kalimat):
    kata_kalimat = kalimat.split()
    for i, kata in enumerate(kata_kalimat):
        if kata.isdigit():
            kata_kalimat[i] = _angka_ke_kata(int(kata))
    kalimat_hasil = " ".join(kata_kalimat)
    return kalimat_hasil