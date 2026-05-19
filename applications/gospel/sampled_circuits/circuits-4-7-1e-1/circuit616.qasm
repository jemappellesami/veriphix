OPENQASM 2.0;
include "qelib1.inc";
qreg q617[4];
cx q617[0],q617[1];
cx q617[1],q617[0];
rz(3*pi/4) q617[1];
cx q617[2],q617[1];
cx q617[1],q617[0];
