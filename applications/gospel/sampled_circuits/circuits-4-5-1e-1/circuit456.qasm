OPENQASM 2.0;
include "qelib1.inc";
qreg q457[4];
rx(pi/4) q457[3];
cx q457[1],q457[2];
cx q457[2],q457[3];
cx q457[2],q457[1];
cx q457[0],q457[1];
