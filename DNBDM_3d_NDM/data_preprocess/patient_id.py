# generate_patient_ids.py

def generate_patient_ids():
    with open("patient_ids.py", "w", encoding="utf-8") as f:
        f.write("patient_ids = [\n")
        for i in range(1, 2401):
            entry = f"'P{i:03d}'"
            if i < 2400:
                if i % 10 == 0:
                    f.write(f"    {entry},\n")
                else:
                    f.write(f"    {entry}, ")
            else:
                f.write(f"    {entry}\n")
        f.write("]\n")

if __name__ == "__main__":
    generate_patient_ids()
    print("✅ patient_ids.py 已生成，包含 P001 → P1200 的完整列表。")
