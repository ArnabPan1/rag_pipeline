import os
class ChunkingStrategy:
    def __init__(self,text_file_path):
        self.text_file_path = text_file_path
    
    def recursive_split(self,text, max_length=1000, overlap=50):
        """Recursively splits text while preserving meaning."""
        if len(text) <= max_length:
            return [text]

        # Try to split at newlines first, then spaces
        split_points = [("\n\n", 2), (" ", 1), ("", 0)]
        
        for sep, extra in split_points:
            parts = text.split(sep)
            if len(parts) > 1:
                chunks = []
                current = ""

                for part in parts:
                    if len(current) + len(part) + extra <= max_length:
                        current += part + sep
                    else:
                        chunks.append(current.strip())
                        current = part

                if current:
                    chunks.append(current.strip())

                # Ensure overlap
                for i in range(1, len(chunks)):
                    chunks[i] = chunks[i-1][-overlap:] + " " + chunks[i]

                return chunks

        return [text[:max_length]]  # Fallback to hard split
    
    def get_pg_no(self,fname):
        fname = fname.replace('.txt','')
        return int(fname.split('_')[-1])

    def get_file_context(self,fname):
        fp = os.apth.join(self.text_file_path,fname)#f'/kaggle/input/gov-land-records/output (2)/{fname}'
        with open(fp, encoding='utf-8') as f:
            return ''.join(f.readlines())
    
    def start_chunking(self):
        files = os.listdir(self.text_file_path)
        pg_no_list = [self.get_pg_no(fname) for fname in files]
        sorted_pg_no_list = sorted(pg_no_list)
        all_res = []
        for i in sorted_pg_no_list:
            fname = f'img_{i}.txt'
            all_res.append(self.get_file_context(fname))
        text = '\n'.join(all_res)
        chunks = self.recursive_split(text)
        return chunks
        
