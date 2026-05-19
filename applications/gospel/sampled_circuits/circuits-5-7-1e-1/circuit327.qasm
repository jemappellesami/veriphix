OPENQASM 2.0;
include "qelib1.inc";
qreg q328[5];
rx(7*pi/4) q328[4];
cx q328[3],q328[4];
cx q328[2],q328[3];
cx q328[1],q328[2];
cx q328[1],q328[0];
