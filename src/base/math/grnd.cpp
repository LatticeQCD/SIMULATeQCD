//
// Created by Lukas Mazur on 08.07.19.
//

#include "grnd.h"




bool RNDHeader::read(std::istream &in, std::string &content) {
    if (in.fail()) {
        rootLogger.error("Could not open file.");
        return false;
    }
    std::string line;

    getline(in, line);
    if (line != "BEGIN_HEADER") {
        rootLogger.error("BEGIN_HEADER not found!");
        return false;
    }

    while (!in.fail()) {
        getline(in, line);
        if (line == "END_HEADER")
            break;
        content.append(line + '\n');
    }
    if (in.fail()) {
        rootLogger.error("END_HEADER not found!");
        return false;
    }
    header_size = in.tellg();
    return true;
}

// called from all nodes, but only root node has already opened file
bool RNDHeader::read(std::istream &in, CommunicationBase &comm) {
    std::string content;
    bool success = true;
    if (comm.IamRoot())
        success = read(in, content);

    if (!comm.single()) {
        comm.root2all(success);
        if (success) {
            comm.root2all(header_size);
            comm.root2all(content);
        }
    }
    if (!success)
        return false;
    std::istringstream str(content);
    return readstream(str, "RNG", true);
}

bool RNDHeader::write(std::ostream &out, CommunicationBase &comm) {
    bool success = true;
    if (comm.IamRoot()) {
        out.precision(10);
        out << "BEGIN_HEADER" << std::endl
            << (*this)
            << "END_HEADER" << std::endl;
        header_size = out.tellp();
        success = !out.fail();
    }
    if (!comm.single()) {
        comm.root2all(header_size);
        comm.root2all(success);
    }
    return success;
}

template<bool onDevice>
void grnd_state<onDevice>::init(){
    memory=MemoryManagement::getMemAt<onDevice>("grndMemory");
    elems = GInd::getLatData().vol4;
    memory->template adjustSize<uint4>(elems);
    state = memory->template getPointer<uint4>();
}


// Here we distribute the random state over the lattice. Each point (NOT link) of the lattice has
// it's own rng state. This is slow but backwards compatible with the old code
template<bool onDevice>
void grnd_state<onDevice>::make_rng_state(unsigned int seed){

    srand48(seed);

    int aux_x, aux_y, aux_z, aux_w;

    std::vector<uint4> temp;
    size_t globalVol = GInd::getLatData().globvol4;

    for (size_t i = 0; i < globalVol; ++i)
    {
        while ( ( aux_x = lrand48() ) <= 128 ) {};
        while ( ( aux_y = lrand48() ) <= 128 ) {};
        while ( ( aux_z = lrand48() ) <= 128 ) {};
        aux_w = lrand48();

        uint4 dummy = {static_cast<unsigned int>(aux_x), static_cast<unsigned int>(aux_y), static_cast<unsigned int>(aux_z), static_cast<unsigned int>(aux_w)};

        //This has to be here, because we need global coordinates!
        int x, y, z, t;
        // THIS ONLY HANDLES EVEN DIMENIONS!
        int par, normInd, tmp;

        divmod(i, globalVol/2, par, normInd);
        normInd = normInd << 0x1;
        // get x,y,z,t
        divmod(normInd, GInd::getLatData().globLX*GInd::getLatData().globLY*GInd::getLatData().globLZ, t, tmp);

        // std::cout << "t coordinate : " << t << std::endl;

        divmod(tmp,     GInd::getLatData().globLX*GInd::getLatData().globLY, z, tmp);
        divmod(tmp,     GInd::getLatData().globLX, y, x);

        // correct effect of divison by two (adjacent odd and even numbers mapped to same number)
        if ( par && !isOdd(x) && !isOdd(y + z + t))
            ++x;
        if (!par && !isOdd(x) &&  isOdd(y + z + t))
            ++x;

        if ((GInd::getLatData().gPosX+ GInd::getLatData().lx > size_t(x)) && (size_t(x) >= GInd::getLatData().gPosX ) &&
            (GInd::getLatData().gPosY+ GInd::getLatData().ly > size_t(y)) && (size_t(y) >= GInd::getLatData().gPosY ) &&
            (GInd::getLatData().gPosZ+ GInd::getLatData().lz > size_t(z)) && (size_t(z) >= GInd::getLatData().gPosZ ) &&
            (GInd::getLatData().gPosT+ GInd::getLatData().lt > size_t(t)) && (size_t(t) >= GInd::getLatData().gPosT )
                ){

            int x_loc, y_loc, z_loc, t_loc;
            x_loc = x - GInd::getLatData().gPosX;
            y_loc = y - GInd::getLatData().gPosY;
            z_loc = z - GInd::getLatData().gPosZ;
            t_loc = t - GInd::getLatData().gPosT;

            gSite site = GInd::getSite(x_loc, y_loc, z_loc, t_loc);

            state[site.isite] = dummy;

            // temp.push_back(dummy);
        }
    }

    // for (size_t i = 0; i < GInd::getLatData().vol4; ++i)
    // {
    //     state[i]=temp[i];
    // }

}


template<bool onDevice>
__host__ __device__  uint4* grnd_state<onDevice>::getElement(gSite site){
    return &state[site.isite];
}

template<bool onDevice>
bool grnd_state<onDevice>::read_header(std::istream &in, CommunicationBase &comm) {
    rootLogger.push_verbosity(OFF);
    if (!header.read(in, comm)){
        rootLogger.error("header.read() failed!");
        return false;
    }
    rootLogger.pop_verbosity();

    bool error = false;
    for (int i = 0; i < 4; i++)
        if (header.dim[i]() != GInd::getLatData().globalLattice()[i]) {
            rootLogger.error("Stored dimension " ,  i ,  " not equal to current lattice size.");
            error = true;
        }

    if (header.dattype() == "RNGSTATE")
        rootLogger.debug("DATATYPE =" ,  header.dattype());
    else {
        rootLogger.error("DATATYPE = " ,  header.dattype() ,  "not recognized.");
        error = true;
    }

    Endianness disken = ENDIAN_AUTO;
    if (header.endian() == "BIG") {
        disken = ENDIAN_BIG;
    } else if (header.endian() == "LITTLE") {
        disken = ENDIAN_LITTLE;
    } else {
        rootLogger.error("Unrecognized ENDIANNESS " ,  header.endian());
        error = true;
    }
    switch_endian = switch_endianness(disken);

    std::stringstream s(header.checksum());
    s >> std::hex >> stored_checksum;
    if (s.fail()) {
        rootLogger.error("Could not interpret checksum " , 
                           header.checksum() ,  "as hexadecimal number.");
        error = true;
    }

    return !error;
}

template<bool onDevice>
bool grnd_state<onDevice>::write_header(Endianness en, std::ostream &out, CommunicationBase &comm) {

    if (en == ENDIAN_AUTO)
        en = get_endianness(false); //use system endianness
    switch_endian = switch_endianness(en);

    for (int mu = 0; mu < 4; mu++)
        header.dim[mu].set(GInd::getLatData().globalLattice()[mu]);

    if (en == ENDIAN_LITTLE)
        header.endian.set(std::string("LITTLE"));
    else
        header.endian.set(std::string("BIG"));

    size_t linktrace;
    stored_checksum = 0;
    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++){

                    gSite site = GInd::getSite(x, y, z, t);
                    uint4 *temp = getElement(site);
                    linktrace = temp->x + temp->y + temp->z + temp->w;
                    stored_checksum += uint32_t(hash_f(linktrace));
            }

    stored_checksum = comm.reduce(stored_checksum);

    std::stringstream s;
    s << std::hex << stored_checksum;
    header.checksum.set(s.str());
    rootLogger.info("Calculated checksum: " ,  s.str());
    return header.write(out, comm);
}

template<bool onDevice>
size_t grnd_state<onDevice>::header_size() {
    return header.size();
}

template<bool onDevice>
bool grnd_state<onDevice>::checksums_match(CommunicationBase &comm) {
        uint32_t checksum = comm.reduce(computed_checksum);
        if (stored_checksum != checksum) {
            rootLogger.error("Checksum mismatch! "
                               ,  std::hex ,  stored_checksum ,  " != "
                               ,  std::hex ,  checksum);
            return false;
        }
        rootLogger.debug("stored_checksum =" ,  std::hex ,  stored_checksum);
        rootLogger.debug("computed_checksum =" ,  std::hex ,  checksum);

        return true;
    }


// writes the rng state to a binary file
template<bool onDevice>
void grnd_state<onDevice>::write_to_file(const std::string &fname, CommunicationBase &comm, Endianness en){


    {
        std::ofstream out;
        if (comm.IamRoot())
            out.open(fname.c_str());
        if (!write_header(en, out, comm)) {
            rootLogger.error("Unable to write RNGSTATE file: " ,  fname);
            return;
        }
    }

    rootLogger.info("Writing RNGSTATE file: " ,  fname);

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();
    const size_t filesize = header_size() + GInd::getLatData().globalLattice().mult() * sizeof(uint4);

    comm.initIOBinary(fname, filesize, sizeof(uint4), header_size(), global, local, WRITE);

    // comm.writeBinary(state, GInd::getLatData().vol4);

    // uint4 buffer[1];
    buffer.resize(GInd::getLatData().vol4);
    size_t index = 0;

    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {

                    gSite site = GInd::getSite(x, y, z, t);

                    buffer[index] = state[site.isite];
                    index++;

                    // comm.writeBinary(buffer, 1);

                }

    process_write_data();

    comm.writeBinary(&buffer[0], GInd::getLatData().vol4);
    comm.closeIOBinary();
}

// reads rng state from binary file
template<bool onDevice>
void grnd_state<onDevice>::read_from_file(const std::string &fname, CommunicationBase &comm){


    {
        std::ifstream in;
        if (comm.IamRoot())
            in.open(fname.c_str());
        if (!read_header(in,comm)){
            throw std::runtime_error(stdLogger.fatal("Error reading header of ", fname.c_str()));
        }
    }

    LatticeDimensions global = GInd::getLatData().globalLattice();
    LatticeDimensions local = GInd::getLatData().localLattice();

    comm.initIOBinary(fname, 0, sizeof(uint4), header_size(), global, local, READ);

    // comm.readBinary(state, GInd::getLatData().vol4);

    // uint4 buffer[1];

    buffer.resize(GInd::getLatData().vol4);
    comm.readBinary(&buffer[0], GInd::getLatData().vol4);

    process_read_data();

    size_t index=0;

    size_t linktrace;
    computed_checksum = 0;

    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {

                    // comm.readBinary(buffer, 1);

                    uint4 dummy = buffer[index];
                    index++;


                    gSite site = GInd::getSite(x, y, z, t);

                    state[site.isite] = dummy;

                    linktrace = dummy.x + dummy.y + dummy.z + dummy.w;
                    computed_checksum += uint32_t(hash_f(linktrace));

                    

                }


    comm.closeIOBinary();

    if (!checksums_match(comm)){
        throw std::runtime_error(stdLogger.fatal("Error checksum!"));
    }
}

// spams random numbers from a local lattice to a text file. Used for testing properties of random number sequence.
template<bool onDevice>
template<class floatT, Layout LatticeLayout, size_t HaloDepth>
void grnd_state<onDevice>::spam_to_file(const CommunicationBase &comm, const LatticeParameters &lp){

    FileWriter Output(comm, lp, "many_rands.rnd", std::ios_base::app);

    for (size_t t = 0; t < GInd::getLatData().lt; t++)
        for (size_t z = 0; z < GInd::getLatData().lz; z++)
            for (size_t y = 0; y < GInd::getLatData().ly; y++)
                for (size_t x = 0; x < GInd::getLatData().lx; x++) {

                    gSite site = GInd::getSite(x, y, z, t);

                    get_rand<floatT>(&state[site.isite]);
                    Output << get_rand<floatT>(&state[site.isite]);
                    Output << "\n";
                }
}

class RNDHeader;

#ifndef CPUONLY
template class grnd_state<true>;
#endif
template class grnd_state<false>;
