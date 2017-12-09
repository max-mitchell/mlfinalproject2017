from ctypes import *
from ctypes.wintypes import *
import win32ui, win32process
import base64
import pdb

class MODULEENTRY32(Structure):
       _fields_ = [ ( 'dwSize' , DWORD ) , 
                ( 'th32ModuleID' , DWORD ),
                ( 'th32ProcessID' , DWORD ),
                ( 'GlblcntUsage' , DWORD ),
                ( 'ProccntUsage' , DWORD ) ,
                ( 'modBaseAddr' , POINTER(DWORD)) ,
                ( 'modBaseSize' , DWORD ) , 
                ( 'hModule' , HMODULE ) ,
                ( 'szModule' , c_char * 256 ),
                ( 'szExePath' , c_char * 260 ) ]

def carrToHex(arr):
	hTmp = [h for h in arr]
	p = 0
	hOut = 0
	for h in hTmp:
		hOut += h*(16**p)
		p += 2
	return hOut


OpenProcess = windll.kernel32.OpenProcess
ReadProcessMemory = windll.kernel32.ReadProcessMemory
GetModuleHandleW = windll.kernel32.GetModuleHandleW


SCORE1 = 0x6aca90
SCORE2 = 0x138
rdScore = ctypes.sizeof(DWORD)

PROCESS_VM_READ = 0x0010
PROCESS_ALL_ACCESS = 0x1F0FFF

TH32CS_SNAPMODULE = 0x00000008

mtry = 0x0b03cde4

#pdb.set_trace()

try:
	pWindow = win32ui.FindWindow(None, u"LUFTRAUSERS").GetSafeHwnd()
except win32ui.error:
	print("Luftrausers not open")
	exit(1)

baseMem = c_uint();
pid = win32process.GetWindowThreadProcessId(pWindow)[1]
snapshot = windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid)
module = MODULEENTRY32()
module.dwSize = ctypes.sizeof(MODULEENTRY32)
if windll.kernel32.Module32First(snapshot, byref(module)):
	print("Read process:", module.szModule)
	baseMem = module.modBaseAddr.contents
else:
	print("error", ctypes.GetLastError())

windll.kernel32.CloseHandle(snapshot)

print("baseMem:", baseMem)

process = OpenProcess(PROCESS_ALL_ACCESS, False, pid);
print("PID:", pid)
if process == 0:
	print("error", ctypes.GetLastError())
	exit(1)

buff = create_string_buffer(4)
buffSize = len(buff)
bytesRead = c_ulonglong(0)

#print((LPCVOID)(baseMem+SCORE1))

if ReadProcessMemory(process, baseMem+SCORE1, buff, buffSize, byref(bytesRead)):
	mInit = carrToHex(buff.raw)
	print("Read", bytesRead.value, "bytes of memory:", mInit)
else:
	print("error", ctypes.GetLastError())

if ReadProcessMemory(process, (LPCVOID)(mInit+SCORE2), buff, buffSize, byref(bytesRead)):
	mSco = carrToHex(buff.raw)
	print("Read", bytesRead.value, "bytes of memory:", mSco)



